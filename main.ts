import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  // Load an image into the canvas
  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  // Convert image to grayscale
  private grayscale(imageData: ImageData): Uint8ClampedArray {
    const gray = new Uint8ClampedArray(imageData.width * imageData.height);
    for (let i = 0; i < imageData.data.length; i += 4) {
      const r = imageData.data[i];
      const g = imageData.data[i + 1];
      const b = imageData.data[i + 2];
      gray[i / 4] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    return gray;
  }

  // Simple thresholding
  private threshold(gray: Uint8ClampedArray, width: number, height: number, t: number = 128): Uint8ClampedArray {
    const bin = new Uint8ClampedArray(width * height);
    for (let i = 0; i < gray.length; i++) {
      bin[i] = gray[i] > t ? 255 : 0;
    }
    return bin;
  }

  // Detect contours (connected white pixels)
  private detectContours(bin: Uint8ClampedArray, width: number, height: number): Point[][] {
    const visited = new Uint8Array(width * height);
    const contours: Point[][] = [];

    const neighbors = [
      [-1, 0], [1, 0], [0, -1], [0, 1],
      [-1, -1], [1, -1], [-1, 1], [1, 1],
    ];

    function inside(x: number, y: number) {
      return x >= 0 && y >= 0 && x < width && y < height;
    }

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (bin[idx] === 255 && !visited[idx]) {
          const stack: Point[] = [{ x, y }];
          const contour: Point[] = [];

          while (stack.length) {
            const p = stack.pop()!;
            const pIdx = p.y * width + p.x;
            if (!inside(p.x, p.y) || visited[pIdx] || bin[pIdx] === 0) continue;
            visited[pIdx] = 1;
            contour.push(p);
            for (const [dx, dy] of neighbors) {
              stack.push({ x: p.x + dx, y: p.y + dy });
            }
          }

          if (contour.length > 5) contours.push(contour);
        }
      }
    }

    return contours;
  }

  // Compute bounding box
  private boundingBox(points: Point[]) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    points.forEach(p => {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    });
    return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
  }

  // Compute polygon vertices using Ramer-Douglas-Peucker
  private approximatePolygon(points: Point[], epsilon: number = 3): Point[] {
    if (points.length < 3) return points;

    function perpendicularDistance(pt: Point, lineStart: Point, lineEnd: Point): number {
      const dx = lineEnd.x - lineStart.x;
      const dy = lineEnd.y - lineStart.y;
      if (dx === 0 && dy === 0) return Math.hypot(pt.x - lineStart.x, pt.y - lineStart.y);
      const t = ((pt.x - lineStart.x) * dx + (pt.y - lineStart.y) * dy) / (dx * dx + dy * dy);
      const projX = lineStart.x + t * dx;
      const projY = lineStart.y + t * dy;
      return Math.hypot(pt.x - projX, pt.y - projY);
    }

    function rdp(pts: Point[], eps: number): Point[] {
      if (pts.length < 3) return pts;
      let maxDist = 0, idx = 0;
      const start = pts[0], end = pts[pts.length - 1];
      for (let i = 1; i < pts.length - 1; i++) {
        const d = perpendicularDistance(pts[i], start, end);
        if (d > maxDist) {
          maxDist = d;
          idx = i;
        }
      }
      if (maxDist > eps) {
        const left = rdp(pts.slice(0, idx + 1), eps);
        const right = rdp(pts.slice(idx), eps);
        return [...left.slice(0, -1), ...right];
      } else return [start, end];
    }

    return rdp(points, epsilon);
  }

  // Classify shape based on number of vertices and area ratio
  private classifyShape(polygon: Point[], contour: Point[]): { type: DetectedShape["type"], confidence: number } {
    const vertices = polygon.length;
    const area = contour.length;
    let type: DetectedShape["type"] = "circle";
    let confidence = 0.8;

    if (vertices === 3) type = "triangle";
    else if (vertices === 4) type = "rectangle";
    else if (vertices === 5) type = "pentagon";
    else if (vertices > 5 && vertices <= 12) type = "star"; // heuristic
    else type = "circle";

    return { type, confidence };
  }

  // Main detection function
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();
    const gray = this.grayscale(imageData);
    const bin = this.threshold(gray, imageData.width, imageData.height, 128);
    const contours = this.detectContours(bin, imageData.width, imageData.height);

    const shapes: DetectedShape[] = [];

    for (const contour of contours) {
      const poly = this.approximatePolygon(contour);
      const { type, confidence } = this.classifyShape(poly, contour);
      const box = this.boundingBox(contour);
      const center = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
      const area = box.width * box.height;

      shapes.push({ type, confidence, boundingBox: box, center, area });
    }

    const processingTime = performance.now() - startTime;

    return { shapes, processingTime, imageWidth: imageData.width, imageHeight: imageData.height };
  }
}
