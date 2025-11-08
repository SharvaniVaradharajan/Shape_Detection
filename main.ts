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

  /**
   * MAIN ALGORITHM TO IMPLEMENT
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   *
   * TODO: Implement shape detection algorithm here
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();

    // TODO: Implement shape detection algorithm
    const shapes: DetectedShape[] = [];

    // Placeholder implementation
    console.log("Shape detection not implemented yet");
    console.log("Image dimensions:", imageData.width, "x", imageData.height);

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: imageData.width,
      imageHeight: imageData.height,
    };
  }

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
}

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
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
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

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});
