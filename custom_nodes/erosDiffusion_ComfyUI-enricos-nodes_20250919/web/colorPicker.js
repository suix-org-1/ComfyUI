import { app } from "../../scripts/app.js";

/**
 * Custom widget for CompositorColorPicker that allows using the browser's EyeDropper API
 */
app.registerExtension({
    name: "ComfyUI.Enrico.ColorPicker",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "CompositorColorPicker") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Override the onNodeCreated method to add our custom widget
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Get references to the RGB input widgets
                const redWidget = this.widgets.find(w => w.name === "red");
                const greenWidget = this.widgets.find(w => w.name === "green");
                const blueWidget = this.widgets.find(w => w.name === "blue");
                const formatWidget = this.widgets.find(w => w.name === "format");
                
                // Create our custom HTML element
                const element = document.createElement("div");
                element.style.display = "flex";
                element.style.alignItems = "center";
                element.style.padding = "5px";
                
                // Create color input
                const colorInput = document.createElement("input");
                colorInput.type = "color";
                colorInput.style.width = "50px";
                colorInput.style.height = "30px";
                colorInput.style.marginRight = "10px";
                colorInput.style.cursor = "pointer";
                colorInput.style.borderRadius = "3px";
                colorInput.style.border = "none";
                
                // Set initial color from the RGB values
                const r = redWidget.value.toString(16).padStart(2, '0');
                const g = greenWidget.value.toString(16).padStart(2, '0');
                const b = blueWidget.value.toString(16).padStart(2, '0');
                colorInput.value = `#${r}${g}${b}`;
                
                // Create eyedropper button
                const eyedropperBtn = document.createElement("button");
                eyedropperBtn.textContent = "🔍";
                eyedropperBtn.title = "Pick color from screen";
                eyedropperBtn.style.cursor = "pointer";
                eyedropperBtn.style.marginRight = "10px";
                eyedropperBtn.style.fontSize = "16px";
                eyedropperBtn.style.padding = "4px 8px";
                eyedropperBtn.style.backgroundColor = "#666";
                eyedropperBtn.style.color = "white";
                eyedropperBtn.style.border = "none";
                eyedropperBtn.style.borderRadius = "3px";
                
                // Create hex color display
                const hexDisplay = document.createElement("span");
                hexDisplay.style.marginLeft = "5px";
                hexDisplay.style.fontFamily = "monospace";
                hexDisplay.style.backgroundColor = "#444";
                hexDisplay.style.padding = "3px 6px";
                hexDisplay.style.borderRadius = "3px";
                hexDisplay.textContent = colorInput.value;
                
                // Function to update RGB widgets when color changes
                const updateWidgetsFromHex = (hexColor) => {
                    // Remove # and parse hex values to RGB
                    const hex = hexColor.substring(1);
                    const r = parseInt(hex.substring(0, 2), 16);
                    const g = parseInt(hex.substring(2, 4), 16);
                    const b = parseInt(hex.substring(4, 6), 16);
                    
                    // Update the widgets
                    redWidget.value = r;
                    greenWidget.value = g;
                    blueWidget.value = b;
                    
                    // Trigger the widget callbacks
                    redWidget.callback(r);
                    greenWidget.callback(g);
                    blueWidget.callback(b);
                    
                    // Update the hex display
                    hexDisplay.textContent = hexColor;
                    
                    // Mark the node as dirty to update visuals
                    this.setDirtyCanvas(true);
                };
                
                // Listen for changes on the color input
                colorInput.addEventListener("input", function() {
                    updateWidgetsFromHex(this.value);
                });
                
                // Eyedropper functionality if supported by the browser
                eyedropperBtn.addEventListener("click", async function() {
                    if ('EyeDropper' in window) {
                        try {
                            const eyeDropper = new EyeDropper();
                            const { sRGBHex } = await eyeDropper.open();
                            colorInput.value = sRGBHex;
                            updateWidgetsFromHex(sRGBHex);
                        } catch (e) {
                            console.log("Eye dropper was cancelled or errored:", e);
                        }
                    } else {
                        alert("EyeDropper API is not supported in your browser");
                    }
                });
                
                // Update color picker when RGB widgets change
                const updateColorFromWidgets = () => {
                    const r = redWidget.value.toString(16).padStart(2, '0');
                    const g = greenWidget.value.toString(16).padStart(2, '0');
                    const b = blueWidget.value.toString(16).padStart(2, '0');
                    const hexColor = `#${r}${g}${b}`;
                    colorInput.value = hexColor;
                    hexDisplay.textContent = hexColor;
                };
                
                // Override the RGB widget callbacks to update our color picker
                const originalRedCallback = redWidget.callback;
                redWidget.callback = function(value) {
                    const result = originalRedCallback?.apply(this, [value]);
                    updateColorFromWidgets();
                    return result;
                };
                
                const originalGreenCallback = greenWidget.callback;
                greenWidget.callback = function(value) {
                    const result = originalGreenCallback?.apply(this, [value]);
                    updateColorFromWidgets();
                    return result;
                };
                
                const originalBlueCallback = blueWidget.callback;
                blueWidget.callback = function(value) {
                    const result = originalBlueCallback?.apply(this, [value]);
                    updateColorFromWidgets();
                    return result;
                };
                
                // Add elements to the container
                element.appendChild(colorInput);
                element.appendChild(eyedropperBtn);
                element.appendChild(hexDisplay);
                
                // Add the DOM widget to the node
                this.addDOMWidget("colorpicker_widget", "colorpicker", element, {
                    serialize: false,
                    hideOnZoom: false,
                });
                
                // Make the original input widgets smaller
                redWidget.computeSize = () => [60, 24];
                greenWidget.computeSize = () => [60, 24]; 
                blueWidget.computeSize = () => [60, 24];
                
                return result;
            };
        }
    },
});