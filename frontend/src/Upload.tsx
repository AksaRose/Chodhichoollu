import React, { useState } from 'react';

const Upload: React.FC = () => {
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
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.status === 'success') {
        setUploadStatus(`File uploaded successfully! Pages processed: ${data.pages}`);
        setSelectedFile(null);
      } else {
        setError(data.error || 'File upload failed.');
        setUploadStatus(null);
      }
    } catch (error: any) {
      setError(`Error uploading file: ${error.message}`);
      setUploadStatus(null);
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