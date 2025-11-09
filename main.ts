import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star" | "polygon";
  confidence: number;
  boundingBox: { x: number; y: number; width: number; height: number };
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

  // Load image into canvas and get ImageData
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

  // Main shape detection
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;

    // -------------------------------
    // Step 1: Convert to Grayscale
    // -------------------------------
    const gray: number[][] = Array.from({ length: height }, () => Array(width).fill(0));
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const r = data[i],
          g = data[i + 1],
          b = data[i + 2];
        gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b;
      }
    }

    // -------------------------------
    // Step 2: Edge Detection (Sobel)
    // -------------------------------
    const edges: number[][] = Array.from({ length: height }, () => Array(width).fill(0));
    const sobelX = [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ];
    const sobelY = [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1],
    ];

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let gx = 0,
          gy = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            gx += gray[y + ky][x + kx] * sobelX[ky + 1][kx + 1];
            gy += gray[y + ky][x + kx] * sobelY[ky + 1][kx + 1];
          }
        }
        const mag = Math.sqrt(gx * gx + gy * gy);
        edges[y][x] = mag > 50 ? 255 : 0; // lowered threshold
      }
    }

    // -------------------------------
    // Step 3: Flood Fill to detect connected components
    // -------------------------------
    const visited = Array.from({ length: height }, () => Array(width).fill(false));
    const shapes: DetectedShape[] = [];
    const directions = [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
      [1, 1],
      [1, -1],
      [-1, 1],
      [-1, -1],
    ];
    const inBounds = (x: number, y: number) => x >= 0 && x < width && y >= 0 && y < height;

    function floodFill(x: number, y: number): [number, number][] {
      const stack: [number, number][] = [[x, y]];
      const points: [number, number][] = [];

      while (stack.length > 0) {
        const [cx, cy] = stack.pop()!;
        if (!inBounds(cx, cy) || visited[cy][cx] || edges[cy][cx] === 0) continue;
        visited[cy][cx] = true;
        points.push([cx, cy]);
        for (const [dx, dy] of directions) stack.push([cx + dx, cy + dy]);
      }

      return points;
    }

    // -------------------------------
    // Step 4: Sort points around centroid
    // -------------------------------
    function sortRegionPoints(points: [number, number][]): [number, number][] {
      const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
      const cy = points.reduce((sum, p) => sum + p[1], 0) / points.length;
      return points.slice().sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
    }

    // -------------------------------
    // Step 5: Ramer-Douglas-Peucker for vertex simplification
    // -------------------------------
    function rdp(points: [number, number][], epsilon: number): [number, number][] {
      if (points.length < 3) return points;

      const lineDist = (p: [number, number], a: [number, number], b: [number, number]) => {
        const [x0, y0] = p,
          [x1, y1] = a,
          [x2, y2] = b;
        return Math.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / Math.hypot(y2 - y1, x2 - x1);
      };

      let maxDist = 0,
        index = 0;
      for (let i = 1; i < points.length - 1; i++) {
        const d = lineDist(points[i], points[0], points[points.length - 1]);
        if (d > maxDist) {
          maxDist = d;
          index = i;
        }
      }

      if (maxDist > epsilon) {
        const left = rdp(points.slice(0, index + 1), epsilon);
        const right = rdp(points.slice(index), epsilon);
        return [...left.slice(0, -1), ...right];
      } else {
        return [points[0], points[points.length - 1]];
      }
    }

    // -------------------------------
    // Step 6: Analyze region for shape
    // -------------------------------
    function analyzeRegion(regionPoints: [number, number][]): DetectedShape {
      const xs = regionPoints.map((p) => p[0]);
      const ys = regionPoints.map((p) => p[1]);
      const minX = Math.min(...xs),
        maxX = Math.max(...xs),
        minY = Math.min(...ys),
        maxY = Math.max(...ys);
      const widthBox = maxX - minX,
        heightBox = maxY - minY;
      const centerX = (minX + maxX) / 2,
        centerY = (minY + maxY) / 2;
      const area = regionPoints.length;

      // Approximate perimeter
      let perimeter = 0;
      const orderedPoints = sortRegionPoints(regionPoints);
      for (let i = 0; i < orderedPoints.length; i++) {
        const [x1, y1] = orderedPoints[i];
        const [x2, y2] = orderedPoints[(i + 1) % orderedPoints.length];
        perimeter += Math.hypot(x2 - x1, y2 - y1);
      }

      // Circularity
      const circularity = (4 * Math.PI * area) / (perimeter * perimeter);

      // Vertices
      const vertices = rdp(orderedPoints, 0.02 * orderedPoints.length).length;

      // Determine type
      let type: DetectedShape["type"] = "polygon";
      let confidence = 0.6;

      if (circularity > 0.6 && vertices <= 5) {
        type = "circle";
        confidence = Math.min(circularity, 0.95);
      } else if (vertices === 3) {
        type = "triangle";
        confidence = 0.9;
      } else if (vertices === 4) {
        const aspectRatio = widthBox / heightBox;
        type = aspectRatio > 0.9 && aspectRatio < 1.1 ? "rectangle" : "rectangle";
        confidence = 0.9;
      } else if (vertices === 5) {
        type = "pentagon";
        confidence = 0.85;
      } else if (vertices >= 10) {
        type = "star";
        confidence = 0.8;
      }

      return {
        type,
        confidence,
        boundingBox: { x: minX, y: minY, width: widthBox, height: heightBox },
        center: { x: centerX, y: centerY },
        area,
      };
    }

    // -------------------------------
    // Step 7: Loop through image to detect regions
    // -------------------------------
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (!visited[y][x] && edges[y][x] > 0) {
          const regionPoints = floodFill(x, y);
          if (regionPoints.length > 10) shapes.push(analyzeRegion(regionPoints));
        }
      }
    }

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: width,
      imageHeight: height,
    };
  }
}

// -------------------------------
// ShapeDetectionApp - UI & Evaluation
// -------------------------------
class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById("originalCanvas") as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById("testImages") as HTMLDivElement;
    this.evaluateButton = document.getElementById("evaluateButton") as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById("evaluationResults") as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(this.detector, this.evaluateButton, this.evaluationResultsDiv);

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) await this.processImage(file);
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";
      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);
      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;
    let html = `<p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>`;
    html += `<p><strong>Shapes Found:</strong> ${shapes.length}</p>`;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `<li>
          <strong>${shape.type}</strong><br>
          Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
          Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(1)})<br>
          Area: ${shape.area.toFixed(1)}px²
        </li>`;
      });
      html += "</ul>";
    } else html += "<p>No shapes detected.</p>";

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();
      let html =
        '<h4>Click to upload your own image or use test images for detection:</h4><div class="test-images-grid">';
      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        html += `<div class="test-image-item" onclick="loadTestImage('${imageName}','${dataUrl}')">
          <img src="${dataUrl}" alt="${imageName}">
          <div>${imageName}</div>
        </div>`;
      });
      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        const response = await fetch(dataUrl);
        const blob = await response.blob();
        const file = new File([blob], name, { type: "image/png" });
        const imageData = await this.detector.loadImage(file);
        const results = await this.detector.detectShapes(imageData);
        this.displayResults(results);
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = "<p>Test images not available.</p>";
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});
