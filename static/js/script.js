/**
 * Function to preview the selected image before uploading
 */
function previewImage(event) {
    const preview = document.getElementById('image-preview');
    const previewContainer = document.getElementById('preview-container');
    preview.src = URL.createObjectURL(event.target.files[0]);
    previewContainer.style.display = 'block';
} 