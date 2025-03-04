import gradio as gr
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# limit the number of channels to display. Must be less than 256
LIMIT = 200

# Load the model
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

model.to(DEVICE)
model.eval()

# Get all layer names
layer_names = []
layer_dict = {}

# Extract all feature extraction layers (conv, relu, maxpool)
features = model.features
for i, layer in enumerate(features):
    layer_type = layer.__class__.__name__
    if 'Conv2d' in layer_type or 'MaxPool2d' in layer_type or 'ReLU' in layer_type:
        name = f"{layer_type.lower()}_{i}"
        layer_names.append(name)
        layer_dict[name] = i

# Define the preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Hook for extracting feature maps
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_feature_maps(img_path, layer_name):
    """Extract feature maps from the specified layer of VGG."""
    # Clear previous activations
    activation.clear()
    
    # Register hook for the selected layer
    layer_index = layer_dict[layer_name]
    handle = features[layer_index].register_forward_hook(get_activation(layer_name))
    
    # Load and preprocess the image
    img = Image.open(img_path).convert('RGB')
    original_img = np.array(img)
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        input_batch = input_batch.to(DEVICE)
        model.features(input_batch)
    
    # Get the feature maps
    feature_maps = activation[layer_name][0].cpu().numpy()
    
    # Remove the hook
    handle.remove()
    
    return feature_maps, original_img

def compute_gram_matrix(feature_maps):
    """Compute the Gram matrix for the given feature maps."""
    # Reshape feature maps
    n_channels, height, width = feature_maps.shape
    reshaped_maps = feature_maps.reshape(n_channels, height * width)
    
    # Compute Gram matrix (correlation between feature maps)
    gram_matrix = np.matmul(reshaped_maps, reshaped_maps.T)
    
    # Normalize by the number of elements
    gram_matrix = gram_matrix / (height * width)
    
    return gram_matrix

def visualize_feature_maps(img_path, layer_name, channel_index=None, view_all=True, channel_offset=0):
    """Visualize feature maps from a layer."""
    feature_maps, original_img = get_feature_maps(img_path, layer_name)
    
    # Get the number of channels
    n_channels = feature_maps.shape[0]
    
    if view_all:
        # Set fixed number of columns to 5
        original_n_channels = n_channels
        if channel_offset > n_channels:
            channel_offset = 0
        elif n_channels - channel_offset > LIMIT:
            n_channels = LIMIT
        else:
            n_channels = n_channels - channel_offset

        n_cols = 5
        n_rows = int(np.ceil(n_channels / n_cols))
        
        # Create a figure with 5 images per row
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        fig.suptitle(f'Feature Maps for Layer: {layer_name} (Total: {original_n_channels} channels)', fontsize=16)
        
        # Flatten the axes array for easier iteration
        axes = axes.flatten() if n_rows > 1 else np.array([axes]).flatten()
        
        # Plot each channel
        for i in range(n_channels):
            # Normalize the feature map for better visualization
            feature_map = feature_maps[i+channel_offset]
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Channel {i+channel_offset}')
            axes[i].axis('off')
        
        # Hide any unused subplots
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')
            
    else:
        # Ensure channel_index is valid
        if channel_index is None or channel_index >= n_channels:
            channel_index = 0
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        ax1.imshow(original_img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Display feature map
        feature_map = feature_maps[channel_index]
        # Normalize the feature map for better visualization
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        im = ax2.imshow(feature_map, cmap='viridis')
        ax2.set_title(f'Layer: {layer_name}, Channel: {channel_index}')
        ax2.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the figure title
    return fig, n_channels

def visualize_gram_matrix(img_path, layer_name, max_channels=None):
    """Visualize the Gram matrix for the selected layer."""
    feature_maps, _ = get_feature_maps(img_path, layer_name)
    
    # Limit the number of channels for visualization if specified
    if max_channels is not None and max_channels > 0 and max_channels < feature_maps.shape[0]:
        feature_maps = feature_maps[:max_channels]
    
    # Compute the Gram matrix
    gram_matrix = compute_gram_matrix(feature_maps)
    
    # Create a figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display the Gram matrix
    gram_matrix_img = ax.imshow(gram_matrix, cmap='viridis')
    ax.set_title(f'Gram Matrix for Layer: {layer_name}\nSize: {gram_matrix.shape[0]}×{gram_matrix.shape[1]}')
    
    # Add colorbar
    plt.colorbar(gram_matrix_img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig, gram_matrix.shape[0]

def update_channel_slider(layer_name, img_path):
    """Update channel slider maximum based on selected layer."""
    if img_path is None:
        return gr.update(maximum=0, value=0)
    
    feature_maps, _ = get_feature_maps(img_path, layer_name)
    n_channels = feature_maps.shape[0]
    return gr.update(maximum=n_channels-1, value=0)

def update_max_channels_slider(layer_name, img_path):
    """Update max channels slider based on selected layer."""
    if img_path is None:
        return gr.update(maximum=100, value=16)
    
    feature_maps, _ = get_feature_maps(img_path, layer_name)
    n_channels = feature_maps.shape[0]
    default_value = min(16, n_channels)  # Default to 50 or the max number of channels
    return gr.update(maximum=n_channels, value=default_value)

def visualize_all(img_path, layer_name, channel_offset):
    """Visualize all channels in a layer."""
    fig, _ = visualize_feature_maps(img_path, layer_name, view_all=True, channel_offset=channel_offset)
    return fig

def visualize_single(img_path, layer_name, channel_index):
    """Visualize a single channel in a layer."""
    fig, _ = visualize_feature_maps(img_path, layer_name, channel_index, view_all=False)
    return fig

def visualize_gram(img_path, layer_name, max_channels):
    """Visualize the Gram matrix for a layer."""
    fig, _ = visualize_gram_matrix(img_path, layer_name, max_channels)
    return fig

# Create Gradio Interface
with gr.Blocks(title="VGG Feature Map Visualizer (PyTorch)") as app:
    gr.Markdown("# VGG Feature Map Visualizer (PyTorch)")
    gr.Markdown("Upload an image and select a layer to visualize its feature maps")
    
    with gr.Tab("All Channels View"):
        with gr.Column():
            # Input controls
            input_image1 = gr.Image(type="filepath", label="Input Image")
            layer_dropdown1 = gr.Dropdown(choices=layer_names, label="Select Layer", value=layer_names[1])
            visualize_button1 = gr.Button("Visualize All Channels")
            channel_offset_slider = gr.Slider(minimum=0, maximum=256, step=1, label="Offset", value=0)

            # Output visualization
            output_plot1 = gr.Plot(label="All Feature Maps (5 per row)")
    
        visualize_button1.click(
            fn=visualize_all,
            inputs=[input_image1, layer_dropdown1, channel_offset_slider],
            outputs=[output_plot1]
        )
    
    with gr.Tab("Single Channel View"):
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                input_image2 = gr.Image(type="filepath", label="Input Image")
                layer_dropdown2 = gr.Dropdown(choices=layer_names, label="Select Layer", value=layer_names[1])
                channel_slider = gr.Slider(minimum=0, maximum=64, step=1, label="Channel Index", value=0)
                visualize_button2 = gr.Button("Visualize Selected Channel")
            
            with gr.Column(scale=2):
                # Output visualization
                output_plot2 = gr.Plot(label="Single Channel Feature Map")
        
        # Update channel slider when layer changes
        layer_dropdown2.change(
            fn=update_channel_slider,
            inputs=[layer_dropdown2, input_image2],
            outputs=[channel_slider]
        )
        
        # Update channel slider when image changes
        input_image2.change(
            fn=update_channel_slider,
            inputs=[layer_dropdown2, input_image2],
            outputs=[channel_slider]
        )
        
        # Visualize the selected channel
        visualize_button2.click(
            fn=visualize_single,
            inputs=[input_image2, layer_dropdown2, channel_slider],
            outputs=[output_plot2]
        )
    
    with gr.Tab("Gram Matrix View"):
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                input_image3 = gr.Image(type="filepath", label="Input Image")
                layer_dropdown3 = gr.Dropdown(choices=layer_names, label="Select Layer", value=layer_names[1])
                max_channels_slider = gr.Slider(minimum=10, maximum=100, step=10, label="Max Channels", value=16)
                visualize_button3 = gr.Button("Visualize Gram Matrix")
            
            with gr.Column(scale=2):
                # Output visualization
                output_plot3 = gr.Plot(label="Gram Matrix Visualization")
        
        # Update max channels slider when layer changes
        layer_dropdown3.change(
            fn=update_max_channels_slider,
            inputs=[layer_dropdown3, input_image3],
            outputs=[max_channels_slider]
        )
        
        # Update max channels slider when image changes
        input_image3.change(
            fn=update_max_channels_slider,
            inputs=[layer_dropdown3, input_image3],
            outputs=[max_channels_slider]
        )
        
        # Visualize the Gram matrix
        visualize_button3.click(
            fn=visualize_gram,
            inputs=[input_image3, layer_dropdown3, max_channels_slider],
            outputs=[output_plot3]
        )

# Launch the app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
