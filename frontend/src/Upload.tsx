import React, { useState } from 'react';

interface UploadProps {
  onFileUploadSuccess: (fileName: string) => void;
}

const Upload: React.FC<UploadProps> = ({ onFileUploadSuccess }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setUploadStatus(null);
      setError(null);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setUploadStatus('Uploading...');
      const response = await fetch('http://localhost:8001/upload', {
        method: 'POST',
        body: formData,
      });

      console.log('Upload response status:', response.status); // Add this line

      if (!response.ok) {
        const errorText = await response.text(); // Try to get more error details
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const data = await response.json();
      if (data.status === 'success') {
        setUploadStatus(`File uploaded successfully! Pages processed: ${data.pages}`);
        setSelectedFile(null);
        onFileUploadSuccess(selectedFile.name); // Pass the file name to the parent component
      } else {
        setError(data.error || 'File upload failed.');
        setUploadStatus(null);
      }
    } catch (error: any) {
      setError(`Error uploading file: ${error.message}`);
      setUploadStatus(null);
      console.error('Detailed upload error:', error); // Modify this line
    }
  };

  return (
    <div style={{ padding: '20px', border: '1px solid #eee', borderRadius: '8px', marginBottom: '20px', backgroundColor: '#f9f9f9' }}>
      <h2 style={{ textAlign: 'center', color: '#333' }}>Upload Document</h2>
      <input
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
        style={{ display: 'block', margin: '10px auto' }}
      />
      <button
        onClick={handleFileUpload}
        disabled={!selectedFile}
        style={{
          display: 'block',
          margin: '10px auto',
          padding: '10px 20px',
          backgroundColor: '#28a745',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          opacity: selectedFile ? 1 : 0.6,
        }}
      >
        Upload PDF
      </button>
      {uploadStatus && <p style={{ textAlign: 'center', color: '#28a745' }}>{uploadStatus}</p>}
      {error && <p style={{ textAlign: 'center', color: '#dc3545' }}>{error}</p>}
    </div>
  );
};

export default Upload;