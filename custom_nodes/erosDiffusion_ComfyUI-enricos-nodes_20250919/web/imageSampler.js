import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Custom widget for ImageColorSampler that allows clicking on an image to sample colors
 */
app.registerExtension({
    name: "ComfyUI.Enrico.ImageSampler",
    
    async setup() {
        // Listen for image_sampler_init event from Python
        api.addEventListener("image_sampler_init", (event) => {
            const detail = event.detail;
            const node = app.graph.getNodeById(detail.node);
            if (!node) return;
            
            // Forward the data to the node instance
            if (node.onImageSamplerInit) {
                node.onImageSamplerInit(detail.data);
            }
        });
        
        // Listen for image_sampler_update event from Python
        api.addEventListener("image_sampler_update", (event) => {
            const detail = event.detail;
            const node = app.graph.getNodeById(detail.node);
            if (!node) return;
            
            // Update the widget with new value
            const widget = node.widgets.find(w => w.name === detail.widget_name);
            if (widget) {
                widget.value = detail.value;
                app.graph.setDirtyCanvas(true);
                
                // Run the workflow again to continue processing
                app.queuePrompt(0, 1); // Continue the workflow
            }
        });
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "ImageColorSampler") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Override the onNodeCreated method to add our custom widget
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Get references to the input widgets
                const samplePointsWidget = this.widgets.find(w => w.name === "sample_points");
                const paletteSizeWidget = this.widgets.find(w => w.name === "palette_size");
                const sampleSizeWidget = this.widgets.find(w => w.name === "sample_size");
                const waitForInputWidget = this.widgets.find(w => w.name === "wait_for_input");
                
                // Hide the sample_points widget as it's just for data storage
                if (samplePointsWidget) {
                    samplePointsWidget.computeSize = () => [0, -4];
                }
                
                // Create main container for our custom widget
                const container = document.createElement("div");
                container.style.width = "auto";
                container.style.height = "auto";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.padding = "10px";
                container.style.resize = "none";
                container.style.overflow = "hidden";
                
                // Create image container
                const imageContainer = document.createElement("div");
                imageContainer.style.backgroundColor = "#333";
                imageContainer.style.border = "1px solid #666";
                imageContainer.style.borderRadius = "4px";
                imageContainer.style.cursor = "crosshair";
                imageContainer.style.display = "block";
                imageContainer.style.width = "auto";
                imageContainer.style.margin = "0 auto";
                imageContainer.style.resize = "none";
                imageContainer.style.overflow = "hidden";
                
                // Create canvas for image display and interaction
                const canvas = document.createElement("canvas");
                canvas.width = 512;  // Initial size, will be updated when image loads
                canvas.height = 512;
                imageContainer.appendChild(canvas);
                
                // Create debug info element to show coordinates
                const debugInfo = document.createElement("div");
                debugInfo.style.backgroundColor = "rgba(0,0,0,0.5)";
                debugInfo.style.color = "#fff";
                debugInfo.style.padding = "5px";
                debugInfo.style.borderRadius = "3px";
                debugInfo.style.fontSize = "12px";
                debugInfo.style.margin = "5px";
                debugInfo.style.display = "block"; // Set to "none" to hide in production
                container.appendChild(debugInfo);
                
                // Create info panel
                const infoPanel = document.createElement("div");
                infoPanel.style.marginTop = "10px";
                infoPanel.style.padding = "8px";
                infoPanel.style.backgroundColor = "#333";
                infoPanel.style.borderRadius = "4px";
                infoPanel.style.fontSize = "12px";
                infoPanel.style.color = "#ccc";
                infoPanel.innerHTML = "Click on image to add color samples<br>Drag points to move<br>CTRL+Click to remove a point<br>Adjust sample size to average colors<br>Click 'Continue Workflow' to proceed";
                
                // Create buttons container
                const buttonsContainer = document.createElement("div");
                buttonsContainer.style.marginTop = "10px";
                buttonsContainer.style.display = "flex";
                buttonsContainer.style.gap = "10px";
                
                // Create clear button
                const clearButton = document.createElement("button");
                clearButton.textContent = "Clear Samples";
                clearButton.style.padding = "6px 12px";
                clearButton.style.backgroundColor = "#555";
                clearButton.style.color = "white";
                clearButton.style.border = "none";
                clearButton.style.borderRadius = "4px";
                clearButton.style.cursor = "pointer";
                
                // Create continue button
                const continueButton = document.createElement("button");
                continueButton.textContent = "Continue Workflow";
                continueButton.style.padding = "6px 12px";
                continueButton.style.backgroundColor = "#3a88fe";
                continueButton.style.color = "white";
                continueButton.style.border = "none";
                continueButton.style.borderRadius = "4px";
                continueButton.style.cursor = "pointer";
                continueButton.style.marginLeft = "auto";
                
                // Add hover effect for buttons
                [clearButton, continueButton].forEach(button => {
                    button.addEventListener("mouseover", function() {
                        this.style.opacity = "0.8";
                    });
                    button.addEventListener("mouseout", function() {
                        this.style.opacity = "1";
                    });
                });
                
                // Add buttons to container
                buttonsContainer.appendChild(clearButton);
                buttonsContainer.appendChild(continueButton);
                
                // Add elements to container in the correct order
                container.appendChild(imageContainer);
                container.appendChild(infoPanel);
                container.appendChild(buttonsContainer);
                
                // Sample points data
                let samplePoints = [];
                const pointSize = 5; // Radius of sample points
                
                // Canvas context and state
                const ctx = canvas.getContext("2d");
                let image = null;
                let imageBase64 = null;
                let selectedPoint = -1;
                let isDragging = false;
                let nodeId = null;
                
                // Store actual dimensions of the original image
                let originalImageWidth = 0;
                let originalImageHeight = 0;
                
                // Method to handle data from Python
                this.onImageSamplerInit = (data) => {
                    if (!data) return;
                    
                    // Store node ID for API calls
                    nodeId = data.node_id;
                    
                    // Load points if any
                    if (data.sample_points && Array.isArray(data.sample_points)) {
                        samplePoints = data.sample_points;
                    }
                    
                    // Update sample size if provided
                    if (data.sample_size && sampleSizeWidget) {
                        sampleSizeWidget.value = data.sample_size;
                    }
                    
                    // Load image if provided
                    if (data.image) {
                        imageBase64 = data.image;
                        loadImageFromBase64(data.image);
                    }
                };
                
                // Load and display image from base64
                const loadImageFromBase64 = (base64Data) => {
                    const img = new Image();
                    img.onload = () => {
                        // Set canvas size to exactly match the image dimensions
                        originalImageWidth = img.width;
                        originalImageHeight = img.height;
                        
                        // Set canvas dimensions to match the image exactly
                        canvas.width = img.width;
                        canvas.height = img.height;
                        
                        // Adjust container size to fit the image exactly
                        imageContainer.style.width = img.width + "px";
                        imageContainer.style.height = img.height + "px";
                        
                        // Draw the image at 1:1 pixel ratio
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0);
                        
                        // Store image reference
                        image = img;
                        
                        // Draw sample points if any
                        drawSamplePoints();
                    };
                    
                    img.src = base64Data;
                };
                
                // Function to continue workflow
                const continueWorkflow = () => {
                    if (!nodeId) return;
                    
                    // Send data back to server to continue the workflow
                    api.fetchApi("/image_sampler/continue", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            node_id: nodeId,
                            sample_points: samplePoints
                        })
                    }).catch(err => console.error("Error continuing workflow:", err));
                };
                
                // Draw sample points
                const drawSamplePoints = () => {
                    if (!ctx || !canvas.width) return;
                    
                    // Redraw the image
                    if (image) {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                    }
                    
                    // Draw each sample point
                    samplePoints.forEach((point, index) => {
                        // Convert normalized coordinates to canvas pixels
                        const x = Math.round(point.x * canvas.width);
                        const y = Math.round(point.y * canvas.height);
                        
                        // Draw outer circle
                        ctx.beginPath();
                        ctx.arc(x, y, pointSize + 2, 0, Math.PI * 2);
                        ctx.fillStyle = "black";
                        ctx.fill();
                        
                        // Draw inner circle with sampled color
                        ctx.beginPath();
                        ctx.arc(x, y, pointSize, 0, Math.PI * 2);
                        ctx.fillStyle = point.color || "#ffffff";
                        ctx.fill();
                        
                        // Highlight selected point
                        if (index === selectedPoint) {
                            ctx.beginPath();
                            ctx.arc(x, y, pointSize + 4, 0, Math.PI * 2);
                            ctx.strokeStyle = "yellow";
                            ctx.lineWidth = 2;
                            ctx.stroke();
                        }
                        
                        // Display color information next to the point with a fixed-size background
                        const hexColor = point.color || "#ffffff";
                        
                        // Parse hex to RGB values
                        const r = parseInt(hexColor.substring(1, 3), 16);
                        const g = parseInt(hexColor.substring(3, 5), 16);
                        const b = parseInt(hexColor.substring(5, 7), 16);
                        const rgbText = `(${r}, ${g}, ${b})`;
                        
                        // Fixed position with consistent offset from the point
                        const labelX = x;
                        // Increase separation between point and label
                        const labelY = y + pointSize + 25; 
                        const padding = 8;
                        
                        // Use a more readable font family with fallbacks
                        const fontFamily = "'Segoe UI', Roboto, 'Helvetica Neue', sans-serif";
                        
                        // Draw better looking label background with rounded corners
                        const cornerRadius = 4;
                        ctx.save(); // Save context state before modifications
                        
                        // Fixed width for the background based on max possible text size
                        // "#FFFFFF" + "(255, 255, 255)" with proper padding
                        const fixedLabelWidth = 100; // Fixed width that accommodates all possible values
                        const fixedLabelHeight = 42; // Fixed height for consistent appearance
                        
                        // Create rounded rectangle background for color label
                        ctx.fillStyle = "rgba(0,0,0,0.75)"; // Darker, more opaque background
                        roundRect(
                            ctx, 
                            labelX - fixedLabelWidth/2, // Center the fixed-width box
                            labelY - 15,
                            fixedLabelWidth,
                            fixedLabelHeight,
                            cornerRadius
                        );
                        
                        // Apply text rendering optimizations
                        ctx.textBaseline = "middle";
                        ctx.shadowColor = "rgba(0,0,0,0.5)";
                        ctx.shadowBlur = 3;
                        ctx.shadowOffsetX = 0;
                        ctx.shadowOffsetY = 1;
                        
                        // Draw hex text with improved visibility
                        ctx.font = `bold 13px ${fontFamily}`;
                        ctx.textAlign = "center";
                        ctx.fillStyle = "#ffffff";
                        ctx.fillText(hexColor, labelX, labelY);
                        
                        // Draw RGB text below hex text
                        ctx.font = `11px ${fontFamily}`;
                        ctx.fillStyle = "#cccccc"; // Slightly dimmer for secondary info
                        ctx.fillText(rgbText, labelX, labelY + 16);
                        
                        ctx.restore(); // Restore context to previous state
                    });
                };
                
                // Helper function to create rounded rectangle path
                const roundRect = (context, x, y, width, height, radius) => {
                    if (width < 2 * radius) radius = width / 2;
                    if (height < 2 * radius) radius = height / 2;
                    
                    context.beginPath();
                    context.moveTo(x + radius, y);
                    context.arcTo(x + width, y, x + width, y + height, radius);
                    context.arcTo(x + width, y + height, x, y + height, radius);
                    context.arcTo(x, y + height, x, y, radius);
                    context.arcTo(x, y, x + width, y, radius);
                    context.closePath();
                    context.fill();
                };
                
                // Check if a point is under the cursor
                const getPointAtPosition = (x, y) => {
                    for (let i = samplePoints.length - 1; i >= 0; i--) {
                        const point = samplePoints[i];
                        const canvasPos = { x: point.x * canvas.width, y: point.y * canvas.height };
                        
                        const distance = Math.sqrt(
                            Math.pow(x - canvasPos.x, 2) + Math.pow(y - canvasPos.y, 2)
                        );
                        
                        if (distance <= pointSize * 2) { // Slightly larger hit area for easier selection
                            return i;
                        }
                    }
                    return -1;
                };
                
                // Get pixel color at a specific position
                const getPixelColorAtPosition = (x, y) => {
                    if (!ctx || !image) return "#FFFFFF";
                    
                    try {
                        // Ensure coordinates are integers and within canvas bounds
                        const pixelX = Math.max(0, Math.min(canvas.width - 1, Math.floor(x)));
                        const pixelY = Math.max(0, Math.min(canvas.height - 1, Math.floor(y)));
                        
                        // Save context state
                        ctx.save();
                        
                        // Clear the canvas and draw just the image without any overlays
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                        
                        // Get the exact pixel data - no averaging or interpolation
                        const pixelData = ctx.getImageData(pixelX, pixelY, 1, 1).data;
                        
                        // Restore context to previous state
                        ctx.restore();
                        
                        // Get exact RGB values (as integers)
                        const r = pixelData[0];
                        const g = pixelData[1];
                        const b = pixelData[2];
                        
                        // Format for hex display
                        const rHex = r.toString(16).padStart(2, '0');
                        const gHex = g.toString(16).padStart(2, '0');
                        const bHex = b.toString(16).padStart(2, '0');
                        
                        // Return the hex representation
                        return `#${rHex}${gHex}${bHex}`;
                    } catch (e) {
                        console.error("Error getting pixel color:", e);
                        return "#FFFFFF";
                    }
                };
                
                // Update debug info display
                const updateDebugInfo = (info) => {
                    if (debugInfo) {
                        debugInfo.textContent = info;
                    }
                };
                
                // Handle mouse events on canvas
                canvas.addEventListener("mousedown", (e) => {
                    e.preventDefault(); // Prevent default browser behavior
                    
                    // Use offsetX and offsetY directly for mouse position
                    const mouseX = e.offsetX;
                    const mouseY = e.offsetY;
                    
                    updateDebugInfo(`Mouse: ${mouseX.toFixed(1)}, ${mouseY.toFixed(1)}`);
                    
                    // Check if click is on an existing point
                    const pointIndex = getPointAtPosition(mouseX, mouseY);
                    
                    if (pointIndex >= 0) {
                        if (e.ctrlKey) {
                            // CTRL+click to delete point
                            samplePoints.splice(pointIndex, 1);
                            selectedPoint = -1;
                            updateSamplePointsWidget();
                            drawSamplePoints();
                        } else {
                            // Select point for dragging
                            selectedPoint = pointIndex;
                            isDragging = true;
                            drawSamplePoints();
                        }
                    } else if (mouseX >= 0 && mouseX <= canvas.width && 
                              mouseY >= 0 && mouseY <= canvas.height && image) {
                        // Calculate normalized coordinates for the new point
                        const normalized = { x: mouseX / canvas.width, y: mouseY / canvas.height };
                        
                        // Add new point
                        const newPoint = { 
                            x: normalized.x, 
                            y: normalized.y, 
                            color: getPixelColorAtPosition(mouseX, mouseY) // Sample color immediately
                        };
                        
                        samplePoints.push(newPoint);
                        selectedPoint = samplePoints.length - 1;
                        isDragging = true;
                        
                        updateSamplePointsWidget();
                        drawSamplePoints();
                    }
                });
                
                canvas.addEventListener("mousemove", (e) => {
                    if (!isDragging || selectedPoint < 0) return;
                    
                    e.preventDefault();
                    
                    // Use offsetX and offsetY directly for mouse position
                    const mouseX = e.offsetX;
                    const mouseY = e.offsetY;
                    
                    // Calculate normalized coordinates
                    const normalized = { x: mouseX / canvas.width, y: mouseY / canvas.height };
                    
                    // Update point position with normalized coordinates
                    samplePoints[selectedPoint].x = normalized.x;
                    samplePoints[selectedPoint].y = normalized.y;
                    
                    // Update debug info
                    updateDebugInfo(`Point: (${normalized.x.toFixed(3)}, ${normalized.y.toFixed(3)})`);
                    
                    // First redraw the image to get accurate sampling
                    if (image) {
                        // Redraw the point's area to ensure we're sampling from the original image
                        ctx.save();
                        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                        ctx.restore();
                    }
                    
                    // Update color at new position
                    samplePoints[selectedPoint].color = getPixelColorAtPosition(mouseX, mouseY);
                    
                    // Then draw all sample points
                    drawSamplePoints();
                });
                
                const handleMouseUp = (e) => {
                    if (isDragging && selectedPoint >= 0) {
                        e.preventDefault();
                        isDragging = false;
                        updateSamplePointsWidget();
                    }
                };
                
                // Track mouse position for debugging
                canvas.addEventListener("mousemove", (e) => {
                    if (debugInfo.style.display !== "block") return;
                    
                    // Use offsetX and offsetY directly
                    const mouseX = e.offsetX;
                    const mouseY = e.offsetY;
                    
                    const normalized = { x: mouseX / canvas.width, y: mouseY / canvas.height };
                    updateDebugInfo(
                        `Mouse: ${mouseX.toFixed(1)}, ${mouseY.toFixed(1)} | `
                    );
                });
                
                canvas.addEventListener("mouseup", handleMouseUp);
                canvas.addEventListener("mouseleave", handleMouseUp);
                
                // Update the hidden widget with sample points data
                const updateSamplePointsWidget = () => {
                    if (samplePointsWidget) {
                        samplePointsWidget.value = JSON.stringify(samplePoints);
                        this.setDirtyCanvas(true);
                    }
                };
                
                // Extract hex codes as a list from sample points
                const getHexCodesList = () => {
                    return samplePoints.map(point => point.color);
                };
                
                // Clear all sample points
                clearButton.addEventListener("click", () => {
                    samplePoints = [];
                    selectedPoint = -1;
                    updateSamplePointsWidget();
                    drawSamplePoints();
                });
                
                // Continue workflow button
                continueButton.addEventListener("click", () => {
                    continueWorkflow();
                });
                
                // When the node receives the output from processing
                this.onExecuted = function(output) {
                    if (!output || !output.hasResult) return;
                    
                    // If we have a sampled colors result, update point colors
                    if (output.sampled_colors) {
                        try {
                            const colors = JSON.parse(output.sampled_colors);
                            if (Array.isArray(colors) && colors.length > 0) {
                                // Update point colors
                                colors.forEach((colorData, index) => {
                                    if (index < samplePoints.length) {
                                        samplePoints[index].color = colorData.hex;
                                    }
                                });
                                drawSamplePoints();
                            }
                        } catch (e) {
                            console.error("Error parsing sampled colors:", e);
                        }
                    }
                };
                
                // Handle image input changes
                this.onImageInput = function(inputData) {
                    // Schedule image loading for next frame to ensure DOM is ready
                    if (inputData && inputData.tensor) {
                        setTimeout(() => loadImageToCanvas(inputData), 0);
                    } else if (imageBase64) {
                        // If we have a base64 image from Python, use that
                        setTimeout(() => loadImageFromBase64(imageBase64), 0);
                    }
                };
                
                // Load and display image from tensor data
                const loadImageToCanvas = (imgData) => {
                    if (!imgData) return;
                    
                    // Store original image dimensions
                    originalImageWidth = imgData.width;
                    originalImageHeight = imgData.height;
                    
                    // Create image from tensor data
                    const imgPixels = new Uint8ClampedArray(imgData.data);
                    const imgDataObj = new ImageData(imgPixels, imgData.width, imgData.height);
                    
                    const offscreenCanvas = new OffscreenCanvas(imgData.width, imgData.height);
                    const offCtx = offscreenCanvas.getContext("2d");
                    offCtx.putImageData(imgDataObj, 0, 0);
                    
                    // Create image object from the offscreen canvas
                    offscreenCanvas.convertToBlob().then(blob => {
                        const img = new Image();
                        img.onload = () => {
                            // Set canvas size to exactly match the image dimensions
                            canvas.width = img.width;
                            canvas.height = img.height;
                            
                            // Adjust container size to fit the image exactly
                            imageContainer.style.width = img.width + "px";
                            imageContainer.style.height = img.height + "px";
                            
                            // Draw the image at 1:1 pixel ratio
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.drawImage(img, 0, 0);
                            
                            // Store image reference
                            image = img;
                            
                            // Draw sample points if any
                            drawSamplePoints();
                        };
                        img.src = URL.createObjectURL(blob);
                    });
                };
                
                // Handle window resize to reposition points correctly
                const resizeObserver = new ResizeObserver(() => {
                    if (image) {
                        // Don't allow resizing - maintain original dimensions
                        canvas.width = originalImageWidth;
                        canvas.height = originalImageHeight;
                        imageContainer.style.width = originalImageWidth + "px";
                        imageContainer.style.height = originalImageHeight + "px";
                        
                        // Redraw everything at the original size
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                        drawSamplePoints();
                    }
                });
                
                // Start observing size changes
                resizeObserver.observe(imageContainer);
                
                // Add the DOM widget to the node
                this.addDOMWidget("image_sampler_widget", "image_sampler", container, {
                    serialize: false,
                    hideOnZoom: false,
                    resizable: false
                });
                
                // Make the input widgets smaller
                if (paletteSizeWidget) paletteSizeWidget.computeSize = () => [100, 24];
                if (sampleSizeWidget) paletteSizeWidget.computeSize = () => [100, 24];
                if (waitForInputWidget) waitForInputWidget.computeSize = () => [100, 24];
                
                return result;
            };
            
            // Add method to capture image input data
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, slotIndex, isConnected, link_info, output) {
                const result = onConnectionsChange?.apply(this, arguments);
                
                // Process only when connecting an input and link_info is valid
                if (type === LiteGraph.INPUT && isConnected && link_info && link_info.origin_id) {
                    // Get the linked node
                    const linkedNode = this.graph.getNodeById(link_info.origin_id);
                    if (linkedNode) {
                        const inputSlot = link_info.origin_slot;
                        const outputData = linkedNode.outputs[inputSlot];
                        
                        // Check if this is an image input
                        if (outputData && outputData.type === "IMAGE") {
                            // Access the tensor data if available
                            const tensorData = linkedNode.getOutputData ? linkedNode.getOutputData(inputSlot) : null;
                            
                            if (tensorData && this.onImageInput) {
                                this.onImageInput({ tensor: tensorData });
                            }
                        }
                    }
                } else if (type === LiteGraph.INPUT && !isConnected) {
                    // If image input is disconnected, clear the canvas
                    const widget = this.widgets.find(w => w.name === "image_sampler_widget");
                    if (widget && widget.value) {
                        const canvas = widget.value.querySelector("canvas");
                        if (canvas) {
                            const ctx = canvas.getContext("2d");
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                        }
                    }
                }
                
                return result;
            };
            
            // Ensure the node updates when new image data is available
            const onExecute = nodeType.prototype.onExecute;
            nodeType.prototype.onExecute = function() {
                const result = onExecute?.apply(this, arguments);
                
                // Check if we have image input
                const imageInput = this.getInputData(0);
                if (imageInput && this.onImageInput) {
                    this.onImageInput({ tensor: imageInput });
                }
                
                return result;
            };
        }
    },
});