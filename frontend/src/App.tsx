import { useState, useEffect } from 'react';
import Upload from './Upload.tsx';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';


interface ChatMessage {
  type: 'user' | 'ai';
  content: string;
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [threadId, setThreadId] = useState('default'); // Can be made dynamic if needed
  const [uploadedPdf, setUploadedPdf] = useState<string | null>(null);

  const handleFileUploadSuccess = (fileName: string) => {
    setUploadedPdf(`http://localhost:8001/pdf/${fileName}`);
  };

  const sendMessage = async () => {
    if (input.trim() === '') return;

    const userMessage: ChatMessage = { type: 'user', content: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput('');

    try {
      const response = await fetch('http://localhost:8001/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input, thread_id: threadId }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.text();
      const aiMessage: ChatMessage = { type: 'ai', content: data };
      setMessages((prevMessages) => [...prevMessages, aiMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: ChatMessage = { type: 'ai', content: 'Error: Could not connect to the API.' };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    }
  };

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', width: '1200px', margin: '20px auto', border: '1px solid #ccc', borderRadius: '8px', padding: '15px', display: 'flex', flexDirection: 'column', gap: '15px', height: '950px' }}>
      {/* Upload Section (at the top) */}
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ textAlign: 'center', color: '#333' }}>Upload PDF</h1>
        <Upload onFileUploadSuccess={handleFileUploadSuccess} />
      </div>

      {/* Two-column layout (PDF display and Chat) */}
      <div style={{ display: 'flex', gap: '15px', flex: 1 }}>
        {/* Left column: PDF display */}
        <div style={{ flex: 1, border: '1px solid #eee', borderRadius: '8px', overflow: 'hidden', height: '700px' }}> {/* Adjusted height for PDF viewer */}
          {uploadedPdf ? (
            <iframe src={uploadedPdf} width="100%" height="100%" style={{ border: 'none' }}></iframe>
          ) : (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: '#555' }}>
              No PDF uploaded yet.
            </div>
          )}
        </div>

        {/* Right column: Chat interface */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <h1 style={{ textAlign: 'center', color: '#333' }}>Chat with Document</h1>
          <div style={{ border: '1px solid #eee', overflowY: 'scroll', padding: '10px', marginBottom: '10px', borderRadius: '4px', backgroundColor: '#f9f9f9', height: '500px' }}> {/* Adjusted height for chat window */}
            {messages.map((msg, index) => (
              <div key={index} style={{ marginBottom: '8px', textAlign: msg.type === 'user' ? 'right' : 'left' }}>
                <span style={{
                  display: 'inline-block',
                  padding: '8px 12px',
                  borderRadius: '18px',
                  backgroundColor: msg.type === 'user' ? '#007bff' : '#e2e6ea',
                  color: msg.type === 'user' ? 'white' : '#333',
                  maxWidth: '75%',
                  wordWrap: 'break-word'
                }}>
                  <ReactMarkdown
                  children={msg.content}
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  />
                </span>
              </div>
            ))}
          </div>
          <div style={{ display: 'flex' }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  sendMessage();
                }
              }}
              style={{ flexGrow: 1, padding: '10px', border: '1px solid #ccc', borderRadius: '4px 0 0 4px', outline: 'none' }}
              placeholder="Type your message..."
            />
            <button
              onClick={sendMessage}
              style={{ padding: '10px 15px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '0 4px 4px 0', cursor: 'pointer', outline: 'none' }}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App
