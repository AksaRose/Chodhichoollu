import { useState, useEffect } from 'react';

interface ChatMessage {
  type: 'user' | 'ai';
  content: string;
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [threadId, setThreadId] = useState('default'); // Can be made dynamic if needed

  const sendMessage = async () => {
    if (input.trim() === '') return;

    const userMessage: ChatMessage = { type: 'user', content: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput('');

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input, thread_id: threadId }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const aiMessage: ChatMessage = { type: 'ai', content: data.reply };
      setMessages((prevMessages) => [...prevMessages, aiMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: ChatMessage = { type: 'ai', content: 'Error: Could not connect to the API.' };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    }
  };

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', width: '800px', margin: '20px auto', border: '1px solid #ccc', borderRadius: '8px', padding: '15px' }}>
      <h1 style={{ textAlign: 'center', color: '#333' }}>Chat with AI</h1>
      <div style={{ border: '1px solid #eee', height: '300px', overflowY: 'scroll', padding: '10px', marginBottom: '10px', borderRadius: '4px', backgroundColor: '#f9f9f9' }}>
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
              {msg.content}
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
  );
}

export default App
