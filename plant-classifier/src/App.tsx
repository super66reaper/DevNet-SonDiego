import React, { useState } from 'react';
import axios from 'axios';

const App: React.FC = () => {
  const [image, setImage] = useState<File | null>(null);

  // Handle file input change
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setImage(file);
  };

  // Submit the image to the backend
  const handleSubmit = async () => {
    if (!image) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append('image', image);

    try {
      await axios.post('http://your-backend-url/classify', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      alert('Image sent successfully!');
    } catch (error) {
      console.error("Error uploading the image:", error);
      alert('Failed to send the image.');
    }
  };

  return (
    <div style={{ textAlign: 'center', padding: '20px' }}>
      <h1>Upload Plant Image</h1>

      {/* File input */}
      <input type="file" accept="image/*" onChange={handleFileChange} />
      
      {/* Submit button */}
      <button onClick={handleSubmit} style={{ marginTop: '20px' }}>
        Send Image
      </button>
    </div>
  );
};

export default App;

