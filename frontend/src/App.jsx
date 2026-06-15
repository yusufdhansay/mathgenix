import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  UploadCloud, 
  History, 
  Settings, 
  CheckCircle2, 
  XCircle, 
  Download, 
  Sparkles, 
  Printer,
  FileDown,
  ShieldAlert,
  Cloud,
  Monitor,
  Key
} from 'lucide-react';
import MathRenderer from './components/MathRenderer';

// API Gateway configured dynamically in settings or build-time env

const BLOOMS_CARDS = [
  { level: 'Remember', desc: 'Identify basic facts, mathematical definitions, equations, and formulas.' },
  { level: 'Understand', desc: 'Explain concepts, interpret equations, or convert word problems into diagrams.' },
  { level: 'Apply', desc: 'Solve structured math problems using given variables and standard algorithms.' },
  { level: 'Analyze', desc: 'Break down complex systems, find calculation errors, or compare logic steps.' },
  { level: 'Evaluate', desc: 'Justify solutions, assess efficiency, and critique alternate math methods.' },
  { level: 'Create', desc: 'Build original math models, synthesize formulas, or design custom equations.' }
];

export default function App() {
  // Navigation State
  const [activeTab, setActiveTab] = useState('workspace'); // workspace, history, settings

  // API Gateway State
  const [apiUrl, setApiUrl] = useState(() => {
    const savedUrl = localStorage.getItem('mathgenix_api_url');
    if (savedUrl) return savedUrl;
    
    // Fallback chain
    if (import.meta.env.VITE_API_BASE_URL) {
      return import.meta.env.VITE_API_BASE_URL;
    }
    
    if (typeof window !== 'undefined') {
      const hostname = window.location.hostname;
      if (hostname === 'localhost' || hostname === '127.0.0.1') {
        return `http://127.0.0.1:8000`;
      }
    }
    return '';
  });
  const [apiUrlInput, setApiUrlInput] = useState(apiUrl);

  // Connection & Config State
  const [backendConnected, setBackendConnected] = useState(false);
  const [ollamaConnected, setOllamaConnected] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('llama-3.3-70b-versatile');
  const [ollamaStatusText, setOllamaStatusText] = useState(apiUrl ? 'Checking...' : 'No API URL');

  // Provider State (cloud or local)
  const [provider, setProvider] = useState('groq'); // 'groq' or 'local'
  const [groqModels, setGroqModels] = useState([]);
  const [groqConnected, setGroqConnected] = useState(false);
  const [groqStatus, setGroqStatus] = useState(apiUrl ? 'Checking...' : 'No API URL');
  const [groqHasKey, setGroqHasKey] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [savingKey, setSavingKey] = useState(false);

  // Upload State
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [extractedText, setExtractedText] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [ocrUsed, setOcrUsed] = useState(false);

  // Generation Parameters
  const [selectedBloom, setSelectedBloom] = useState('Apply');
  const [loading, setLoading] = useState(false);
  const [generationError, setGenerationError] = useState('');
  const [currentQuestions, setCurrentQuestions] = useState([]); // Array of generated question objects
  const [generatedTopicsHistory, setGeneratedTopicsHistory] = useState([]); // Track previously generated topics for uniqueness

  // PDF Export Option
  const [includeSolutions, setIncludeSolutions] = useState(true);


  // Saved History State
  const [historyDecks, setHistoryDecks] = useState([]);

  // Fetch health, models, and Groq info on mount + when apiUrl changes + setup polling
  useEffect(() => {
    if (!apiUrl) {
      setBackendConnected(false);
      setOllamaConnected(false);
      setOllamaStatusText('No API URL');
      setGroqConnected(false);
      setGroqStatus('No API URL');
      return;
    }

    checkHealth();
    fetchModels();
    fetchGroqModels();
    
    // Poll health status every 10 seconds to keep diagnostics updated
    const intervalId = setInterval(() => {
      checkHealth();
    }, 10000);

    // Load local history if available
    const saved = localStorage.getItem('math_deck_history');
    if (saved) {
      try {
        setHistoryDecks(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to parse history', e);
      }
    }

    return () => clearInterval(intervalId);
  }, [apiUrl]);

  const checkHealth = async () => {
    if (!apiUrl) return;
    try {
      const response = await fetch(`${apiUrl}/api/health`);
      const data = await response.json();
      setBackendConnected(true);
      // Ollama status
      if (data.ollama && data.ollama.available) {
        setOllamaConnected(true);
        setOllamaStatusText('Ollama Connected');
      } else {
        setOllamaConnected(false);
        setOllamaStatusText('Ollama Offline');
      }
      // Groq status
      if (data.groq) {
        setGroqConnected(data.groq.available);
        setGroqStatus(data.groq.status);
        setGroqHasKey(data.groq.has_key);
      }
    } catch (err) {
      setBackendConnected(false);
      setOllamaConnected(false);
      setOllamaStatusText('API Offline');
      setGroqConnected(false);
      setGroqStatus('API Offline');
    }
  };

  const fetchModels = async () => {
    if (!apiUrl) return;
    try {
      const response = await fetch(`${apiUrl}/api/models`);
      const data = await response.json();
      setModels(data.models || []);
      if (data.models && data.models.length > 0) {
        const qwenTag = data.models.find(m => m.startsWith('qwen2-math'));
        const deepseekTag = data.models.find(m => m.startsWith('deepseek-r1'));
        const wizardTag = data.models.find(m => m.startsWith('wizard-math'));
        const llamaTag = data.models.find(m => m.startsWith('llama3'));

        if (qwenTag) {
          setSelectedModel(qwenTag);
        } else if (deepseekTag) {
          setSelectedModel(deepseekTag);
        } else if (wizardTag) {
          setSelectedModel(wizardTag);
        } else if (llamaTag) {
          setSelectedModel(llamaTag);
        } else {
          setSelectedModel(data.models[0]);
        }
      }
    } catch (err) {
      setModels(['qwen2-math', 'deepseek-r1', 'wizard-math', 'llama3', 'mistral']);
    }
  };

  const fetchGroqModels = async () => {
    if (!apiUrl) return;
    try {
      const response = await fetch(`${apiUrl}/api/groq-models`);
      const data = await response.json();
      setGroqModels(data.models || []);
    } catch (err) {
      setGroqModels([]);
    }
  };

  const saveGroqApiKey = async () => {
    if (!apiUrl) return;
    if (!apiKeyInput.trim()) return;
    setSavingKey(true);
    try {
      const response = await fetch(`${apiUrl}/api/set-groq-key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: apiKeyInput.trim() }),
      });
      const data = await response.json();
      if (data.success) {
        setGroqConnected(data.groq_available);
        setGroqStatus(data.groq_status);
        setGroqHasKey(true);
        setApiKeyInput('');
      }
    } catch (err) {
      console.error('Failed to save API key', err);
    } finally {
      setSavingKey(false);
    }
  };

  const saveApiUrl = () => {
    let formattedUrl = apiUrlInput.trim();
    if (!formattedUrl) return;
    
    // Remove trailing slash if present
    if (formattedUrl.endsWith('/')) {
      formattedUrl = formattedUrl.slice(0, -1);
    }
    
    // Ensure it starts with http:// or https://
    if (!/^https?:\/\//i.test(formattedUrl)) {
      formattedUrl = 'https://' + formattedUrl;
    }
    
    setApiUrl(formattedUrl);
    setApiUrlInput(formattedUrl);
    localStorage.setItem('mathgenix_api_url', formattedUrl);
  };

  // Drag & Drop Handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  // Perform backend file upload and text extraction
  const handleFileUpload = async (uploadedFile) => {
    if (!apiUrl) return;
    setFile(uploadedFile);
    setUploading(true);
    setUploadSuccess(false);
    setUploadError('');
    setGenerationError('');
    setOcrUsed(false);
    setGeneratedTopicsHistory([]); // Reset topic history for new document
    
    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await fetch(`${apiUrl}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMsg = 'Failed to extract text from document';
        try {
          const errorData = await response.json();
          errorMsg = errorData.detail || errorMsg;
        } catch (_) {}
        throw new Error(errorMsg);
      }

      const data = await response.json();
      setExtractedText(data.text);
      setOcrUsed(data.ocr_used || false);
      setUploadSuccess(true);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  };

  // Request backend generation of Bloom's math questions
  const triggerGeneration = async () => {
    if (!apiUrl) return;
    if (!extractedText) return;
    setLoading(true);
    setGenerationError('');
    setCurrentQuestions([]);

    try {
      const response = await fetch(`${apiUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: extractedText,
          taxonomy_level: selectedBloom,
          model_name: selectedModel,
          provider: provider,
          previous_topics: generatedTopicsHistory
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate math questions.');
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      const questionSet = data.questions || [];
      if (questionSet.length === 0) {
        throw new Error('No questions were parsed from the model response.');
      }

      setCurrentQuestions(questionSet);

      // Accumulate topics for deduplication on next generation
      const newTopics = questionSet
        .map(q => q.topic)
        .filter(t => t && t.trim());
      setGeneratedTopicsHistory(prev => [...prev, ...newTopics]);

      // Save to local history
      const newDeck = {
        id: Date.now(),
        filename: file?.name || 'Copied Text',
        timestamp: new Date().toLocaleDateString(),
        bloom: selectedBloom,
        model: selectedModel,
        questions: questionSet
      };

      const updatedHistory = [newDeck, ...historyDecks];
      setHistoryDecks(updatedHistory);
      localStorage.setItem('math_deck_history', JSON.stringify(updatedHistory));

    } catch (err) {
      setGenerationError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadSavedDeck = (deck) => {
    setCurrentQuestions(deck.questions);
    setSelectedBloom(deck.bloom);
    setSelectedModel(deck.model);
    setActiveTab('workspace');
  };

  const deleteDeck = (id, e) => {
    e.stopPropagation();
    const updated = historyDecks.filter(d => d.id !== id);
    setHistoryDecks(updated);
    localStorage.setItem('math_deck_history', JSON.stringify(updated));
  };

  const exportMarkdown = (questions) => {
    let md = `# Math Questions (${selectedBloom} Level)\n\nGenerated from: ${file?.name || 'Workspace Document'}\n\n`;
    questions.forEach((q, idx) => {
      md += `### Question ${idx + 1}: ${q.topic || 'Math Problem'}\n${q.question || ''}\n\n`;
      md += `* **Final Answer:** ${q.answer || ''}\n\n`;
      md += `* **Solution Steps:**\n`;
      const steps = Array.isArray(q.solution_steps) 
        ? q.solution_steps 
        : (typeof q.solution_steps === 'string' ? [q.solution_steps] : []);
      steps.forEach((step, sIdx) => {
        md += `  ${sIdx + 1}. ${step}\n`;
      });
      md += `\n---\n\n`;
    });

    const element = document.createElement("a");
    const fileBlob = new Blob([md], { type: 'text/markdown' });
    element.href = URL.createObjectURL(fileBlob);
    element.download = `math_questions_${selectedBloom.toLowerCase()}.md`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleExportPDF = () => {
    window.print();
  };

  return (
    <div className="app-container">
      {/* Sidebar Section */}
      <aside className="sidebar">
        <div className="logo-section">
          <span className="logo-icon">📚</span>
          <h2 className="logo-text">MathGenix</h2>
        </div>

        <nav className="nav-list">
          <button 
            className={`nav-item ${activeTab === 'workspace' ? 'active' : ''}`}
            onClick={() => setActiveTab('workspace')}
          >
            <Brain />
            <span>Workspace</span>
          </button>
          
          <button 
            className={`nav-item ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            <History />
            <span>Question Decks</span>
          </button>
          
          <button 
            className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            <Settings />
            <span>Server Status</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <div className="ollama-status-badge">
            <span className={`status-dot ${groqConnected ? 'online' : 'offline'}`}></span>
            <span>{groqConnected ? 'Groq Cloud' : 'Groq Offline'}</span>
          </div>
          <div className="ollama-status-badge">
            <span className={`status-dot ${ollamaConnected ? 'online' : 'offline'}`}></span>
            <span>{ollamaStatusText}</span>
          </div>
        </div>
      </aside>

      {/* Main Panel */}
      <main className="main-content">
        
        {/* TAB 1: WORKSPACE */}
        {activeTab === 'workspace' && (
          <div className="screen-only">
            <header>
              <h1>Math Generation Hub</h1>
              <p>Upload study materials and use local LLMs to generate high-quality math assessments.</p>
            </header>

            {!apiUrl && (
              <div className="feedback-box incorrect" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2rem' }}>
                <ShieldAlert size={24} style={{ color: 'var(--accent-rose)', flexShrink: 0 }} />
                <div>
                  <strong style={{ color: 'var(--accent-rose)' }}>Backend API Connection Required:</strong>
                  <p style={{ fontSize: '0.9rem', marginTop: '0.2rem', color: 'var(--text-secondary)' }}>
                    Your frontend is hosted on Vercel, but it does not know where your FastAPI backend is. 
                    Please click the <strong>Server Status</strong> tab on the sidebar and enter your Render backend URL (e.g. <code>https://your-service.onrender.com</code>).
                  </p>
                </div>
              </div>
            )}

            <div className="card">
              <h3 className="card-title">
                <UploadCloud style={{ color: 'var(--accent-primary)' }} />
                <span>1. Load Document</span>
              </h3>
              
              {!apiUrl ? (
                <div style={{ padding: '2.5rem 1.5rem', textAlign: 'center', background: 'rgba(15, 23, 42, 0.4)', borderRadius: '10px', border: '1px dashed var(--glass-border)' }}>
                  <span style={{ fontSize: '2.5rem', display: 'block', marginBottom: '0.75rem' }}>🔌</span>
                  <h4 style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Connection Needed</h4>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.5 }}>
                    Please configure and connect your backend API under the <strong>Server Status</strong> tab first to enable document upload and question generation.
                  </p>
                </div>
              ) : !file ? (
                <div 
                  className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => document.getElementById('file-upload-input').click()}
                >
                  <input 
                    id="file-upload-input" 
                    type="file" 
                    accept=".pdf,.txt,.docx,.jpg,.jpeg,.png,.webp,.heic" 
                    style={{ display: 'none' }} 
                    onChange={handleFileChange}
                  />
                  <span className="upload-icon">📥</span>
                  <h3>Drag & Drop Study Document</h3>
                  <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>
                    Supports PDF, Word, Text, and Image files (JPG, PNG)
                  </p>
                  <button className="upload-btn" type="button">Browse Local Files</button>
                </div>
              ) : (
                <div className="file-info">
                  <div className="file-info-details">
                    <span className="file-icon">📄</span>
                    <div>
                      <p style={{ fontWeight: 600 }}>{file.name}</p>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </div>
                  <button 
                    className="btn-icon" 
                    onClick={() => {
                      setFile(null);
                      setExtractedText('');
                      setUploadSuccess(false);
                      setUploadError('');
                      setOcrUsed(false);
                      setCurrentQuestions([]);
                    }}
                  >
                    <XCircle size={18} style={{ color: 'var(--accent-rose)' }} />
                  </button>
                </div>
              )}

              {uploading && (
                <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className="spinner" style={{ width: '20px', height: '20px', borderWidth: '2px' }}></span>
                  <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Extracting content...</span>
                </div>
              )}

              {uploadSuccess && (
                <p style={{ color: 'var(--accent-emerald)', fontSize: '0.9rem', fontWeight: 500, marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <CheckCircle2 size={16} />
                  {ocrUsed 
                    ? '📷 Text extracted via AI Vision OCR! Ready to generate.' 
                    : 'Document processed successfully! Ready to generate.'
                  }
                </p>
              )}

              {uploadError && (
                <div className="feedback-box incorrect" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '1rem', marginBottom: '0' }}>
                  <XCircle size={20} style={{ color: 'var(--accent-rose)', flexShrink: 0 }} />
                  <div>
                    <strong style={{ color: 'var(--accent-rose)' }}>Upload/Extraction Error:</strong>
                    <p style={{ fontSize: '0.85rem', marginTop: '0.2rem', color: 'var(--text-secondary)' }}>{uploadError}</p>
                  </div>
                </div>
              )}
            </div>

            {uploadSuccess && (
              <div className="card">
                <h3 className="card-title">
                  <Sparkles style={{ color: 'var(--accent-secondary)' }} />
                  <span>2. Customize Blueprint</span>
                </h3>
                
                <div className="form-group">
                  <label className="form-label">Select Bloom's Taxonomy Level</label>
                  <div className="selector-grid">
                    {BLOOMS_CARDS.map(item => (
                      <div 
                        key={item.level} 
                        className={`selector-card ${selectedBloom === item.level ? 'selected' : ''}`}
                        onClick={() => setSelectedBloom(item.level)}
                      >
                        <h4>{item.level}</h4>
                        <p>{item.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">Inference Provider</label>
                  <div className="provider-toggle" style={{ marginTop: '0.5rem' }}>
                    <button 
                      type="button"
                      className={`provider-pill cloud ${provider === 'groq' ? 'active' : ''}`}
                      onClick={() => {
                        setProvider('groq');
                        if (groqModels.length > 0) setSelectedModel(groqModels[0].id);
                      }}
                    >
                      <span className="pill-icon">☁️</span>
                      <span>Cloud (Groq)</span>
                    </button>
                    <button 
                      type="button"
                      className={`provider-pill local ${provider === 'local' ? 'active' : ''}`}
                      onClick={() => {
                        setProvider('local');
                        if (models.length > 0) setSelectedModel(models[0]);
                      }}
                    >
                      <span className="pill-icon">💻</span>
                      <span>Local (Ollama)</span>
                    </button>
                  </div>
                  {provider === 'groq' && !groqHasKey && (
                    <p style={{ fontSize: '0.8rem', color: 'var(--accent-rose)', marginTop: '0.5rem' }}>
                      ⚠️ No Groq API key configured. Go to Settings to add your free key.
                    </p>
                  )}
                  {provider === 'groq' && groqConnected && (
                    <p style={{ fontSize: '0.8rem', color: '#34d399', marginTop: '0.5rem' }}>
                      ✅ Groq connected — ~300 tok/s, 5 questions in seconds
                    </p>
                  )}
                </div>

                <div className="form-group">
                  <label className="form-label">
                    {provider === 'groq' ? 'Cloud Model' : 'Local Reasoning Model'}
                  </label>
                  <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginTop: '0.5rem' }}>
                    {provider === 'groq' ? (
                      groqModels.map(m => {
                        const isSelected = selectedModel === m.id;
                        return (
                          <button
                            key={m.id}
                            type="button"
                            onClick={() => setSelectedModel(m.id)}
                            style={{
                              background: isSelected 
                                ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)' 
                                : 'rgba(15, 23, 42, 0.4)',
                              color: isSelected ? '#ffffff' : 'var(--text-secondary)',
                              border: isSelected ? '1px solid transparent' : '1px solid var(--glass-border)',
                              padding: '0.65rem 1.25rem',
                              borderRadius: '10px',
                              fontWeight: 600,
                              fontSize: '0.9rem',
                              cursor: 'pointer',
                              transition: 'var(--transition-smooth)',
                              boxShadow: isSelected ? '0 0 15px rgba(99, 102, 241, 0.2)' : 'none',
                              display: 'flex',
                              flexDirection: 'column',
                              alignItems: 'flex-start',
                              gap: '0.2rem'
                            }}
                          >
                            <span>⚡ {m.name}</span>
                            <span style={{ fontSize: '0.7rem', fontWeight: 400, opacity: 0.7 }}>{m.description}</span>
                          </button>
                        );
                      })
                    ) : (
                      models.map(m => {
                        const isSelected = selectedModel === m;
                        let displayName = m;
                        if (m.startsWith('qwen2-math')) displayName = '⚡ Qwen-2 Math';
                        else if (m.startsWith('deepseek-r1')) displayName = '🧠 DeepSeek R1';
                        else if (m.startsWith('llama3')) displayName = '🦙 Llama 3';
                        else if (m.startsWith('wizard-math')) displayName = '🧙 WizardMath';
                        else if (m.startsWith('mistral')) displayName = '💨 Mistral';
                        
                        return (
                          <button
                            key={m}
                            type="button"
                            onClick={() => setSelectedModel(m)}
                            style={{
                              background: isSelected 
                                ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)' 
                                : 'rgba(15, 23, 42, 0.4)',
                              color: isSelected ? '#ffffff' : 'var(--text-secondary)',
                              border: isSelected ? '1px solid transparent' : '1px solid var(--glass-border)',
                              padding: '0.65rem 1.25rem',
                              borderRadius: '10px',
                              fontWeight: 600,
                              fontSize: '0.9rem',
                              cursor: 'pointer',
                              transition: 'var(--transition-smooth)',
                              boxShadow: isSelected ? '0 0 15px rgba(16, 185, 129, 0.2)' : 'none',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.4rem'
                            }}
                          >
                            <span>{displayName}</span>
                          </button>
                        );
                      })
                    )}
                  </div>
                </div>


                {generationError && (
                  <div className="feedback-box incorrect" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <XCircle size={20} />
                    <div>
                      <strong>Generation Error:</strong>
                      <p style={{ fontSize: '0.85rem', marginTop: '0.2rem' }}>{generationError}</p>
                    </div>
                  </div>
                )}

                <button 
                  className="btn-generate"
                  disabled={loading || !extractedText}
                  onClick={triggerGeneration}
                >
                  {loading ? (
                    <>
                      <span className="spinner" style={{ width: '20px', height: '20px', borderWidth: '2px' }}></span>
                      <span>Creating Questions...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles size={20} />
                      <span>Generate Questions</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Display Generated Question Set */}
            {currentQuestions.length > 0 && (
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2.5rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '1.25rem' }}>
                  <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.4rem' }}>
                    🎯 Question Set ({selectedBloom})
                  </h3>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    {/* PDF Toggle Options */}
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)', cursor: 'pointer', userSelect: 'none' }}>
                      <input 
                        type="checkbox" 
                        checked={includeSolutions} 
                        onChange={(e) => setIncludeSolutions(e.target.checked)} 
                        style={{ accentColor: 'var(--accent-primary)' }}
                      />
                      <span>Include Solutions in PDF</span>
                    </label>

                    <button 
                      className="btn-secondary" 
                      style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', padding: '0.5rem 1rem' }}
                      onClick={() => exportMarkdown(currentQuestions)}
                    >
                      <Download size={16} />
                      <span>Markdown</span>
                    </button>
                    
                    <button 
                      className="btn-primary" 
                      style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', padding: '0.5rem 1.2rem', background: 'linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%)' }}
                      onClick={handleExportPDF}
                    >
                      <Printer size={16} />
                      <span>Export as PDF</span>
                    </button>
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                  {currentQuestions.map((q, idx) => {
                    const steps = Array.isArray(q.solution_steps) 
                      ? q.solution_steps 
                      : (typeof q.solution_steps === 'string' ? [q.solution_steps] : []);
                    return (
                      <div key={q.id || idx} style={{ background: 'rgba(15, 23, 42, 0.2)', border: '1px solid var(--glass-border)', padding: '1.5rem', borderRadius: '12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                          <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--accent-secondary)', fontWeight: 700 }}>
                            Problem {idx + 1} — {q.topic || 'Math Problem'}
                          </span>

                        </div>
                        <div style={{ fontSize: '1.1rem', margin: '0.75rem 0 1.25rem 0', lineHeight: 1.6 }}>
                          <MathRenderer text={q.question || ''} />
                        </div>
                        
                        <details style={{ cursor: 'pointer' }}>
                          <summary style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', userSelect: 'none' }}>
                            View Solution Blueprint
                          </summary>
                          <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(15, 23, 42, 0.4)', borderRadius: '8px', borderLeft: '3px solid var(--accent-primary)' }}>
                            <p style={{ fontSize: '0.9rem', marginBottom: '0.5rem' }}>
                              <strong>Answer:</strong> <MathRenderer text={q.answer || ''} />
                            </p>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                              <strong style={{ display: 'block', marginBottom: '0.25rem' }}>Solution Steps:</strong>
                              <ol style={{ paddingLeft: '1.25rem' }}>
                                {steps.map((step, sIdx) => (
                                  <li key={sIdx} style={{ marginBottom: '0.4rem' }}>
                                    <MathRenderer text={step || ''} />
                                  </li>
                                ))}
                              </ol>
                            </div>
                          </div>
                        </details>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: SAVED QUESTION DECKS */}
        {activeTab === 'history' && (
          <div className="screen-only">
            <header>
              <h1>Question Decks</h1>
              <p>Review previously generated math sets and reload them into the workspace.</p>
            </header>

            {historyDecks.length === 0 ? (
              <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>No question sets generated yet.</p>
                <button className="btn-primary" onClick={() => setActiveTab('workspace')}>
                  Generate First Set
                </button>
              </div>
            ) : (
              <div className="history-grid">
                {historyDecks.map(deck => (
                  <div 
                    key={deck.id} 
                    className="history-card"
                    onClick={() => loadSavedDeck(deck)}
                  >
                    <div className="history-meta">
                      <span>{deck.timestamp}</span>
                      <span style={{ color: 'var(--accent-secondary)', fontWeight: 600 }}>{deck.bloom}</span>
                    </div>
                    <div className="history-topic">
                      {deck.filename}
                    </div>
                    <p className="history-desc">
                      Generated {deck.questions.length} problems using model: <code>{deck.model}</code>
                    </p>
                    <div className="history-actions">
                      <button 
                        className="btn-primary" 
                        style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem' }}
                        onClick={(e) => {
                          e.stopPropagation();
                          loadSavedDeck(deck);
                        }}
                      >
                        Reload Set
                      </button>
                      <button 
                        className="btn-secondary" 
                        style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem' }}
                        onClick={(e) => {
                          e.stopPropagation();
                          exportMarkdown(deck.questions);
                        }}
                      >
                        <Download size={12} />
                      </button>
                      <button 
                        className="btn-icon" 
                        onClick={(e) => deleteDeck(deck.id, e)}
                        title="Delete set"
                        style={{ marginLeft: 'auto' }}
                      >
                        <XCircle size={14} style={{ color: 'var(--accent-rose)' }} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 3: SERVER HEALTH & SETTINGS */}
        {activeTab === 'settings' && (
          <div className="screen-only">
            <header>
              <h1>Server Status</h1>
              <p>Manage system connections, API keys, and configure platform settings.</p>
            </header>

            {/* API Gateway URL Configuration */}
            <div className="card">
              <h3 className="card-title">🌐 API Gateway Configuration</h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.6, marginBottom: '1rem' }}>
                Set the URL of your FastAPI backend server. 
                {window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1' && (
                  <span style={{ display: 'block', marginTop: '0.5rem', color: 'var(--accent-secondary)' }}>
                    💡 Tip: Since you are running in production on Vercel, configure this to your Render URL (e.g. <code>https://your-app.onrender.com</code>).
                  </span>
                )}
              </p>

              <div style={{ marginTop: '1rem' }}>
                <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span>API Gateway URL</span>
                </label>
                <div className="api-key-input-group">
                  <input 
                    type="text"
                    className="api-key-input"
                    placeholder="https://mathgenix-backend.onrender.com"
                    value={apiUrlInput}
                    onChange={(e) => setApiUrlInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && saveApiUrl()}
                  />
                  <button 
                    className="btn-save-key"
                    onClick={saveApiUrl}
                    disabled={!apiUrlInput.trim()}
                  >
                    Save URL
                  </button>
                </div>
                {apiUrl && (
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                    Active Endpoint: <code style={{ color: 'var(--accent-primary)' }}>{apiUrl}</code>
                  </p>
                )}
                {!apiUrl && (
                  <p style={{ fontSize: '0.8rem', color: 'var(--accent-rose)', marginTop: '0.5rem' }}>
                    ⚠️ No API URL configured. The app will not be able to generate questions.
                  </p>
                )}
              </div>
            </div>

            {/* Groq Cloud Configuration */}
            <div className="card">
              <h3 className="card-title">☁️ Groq Cloud (Ultra-Fast LPU Inference)</h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.6, marginBottom: '1rem' }}>
                Groq's custom LPU chips generate at ~300 tok/s — producing 5 questions in 3–5 seconds.
                Get your free API key at <a href="https://console.groq.com" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-primary)' }}>console.groq.com</a>
              </p>

              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem 0', borderBottom: '1px solid var(--glass-border)' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Groq Cloud Status:</span>
                <span style={{ color: groqConnected ? 'var(--accent-emerald)' : 'var(--accent-rose)', fontWeight: 600 }}>
                  {groqStatus}
                </span>
              </div>

              <div style={{ marginTop: '1rem' }}>
                <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Key size={16} />
                  <span>Groq API Key</span>
                </label>
                <div className="api-key-input-group">
                  <input 
                    type="password"
                    className="api-key-input"
                    placeholder={groqHasKey ? '••••••••••••••••••••• (key saved)' : 'gsk_xxxxxxxxxxxxxxxxxxxxxxxx'}
                    value={apiKeyInput}
                    onChange={(e) => setApiKeyInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && saveGroqApiKey()}
                  />
                  <button 
                    className="btn-save-key"
                    onClick={saveGroqApiKey}
                    disabled={!apiKeyInput.trim() || savingKey}
                  >
                    {savingKey ? 'Saving...' : 'Save Key'}
                  </button>
                </div>
                {groqHasKey && groqConnected && (
                  <p style={{ fontSize: '0.8rem', color: '#34d399', marginTop: '0.5rem' }}>✅ API key verified and active</p>
                )}
                {groqHasKey && !groqConnected && (
                  <p style={{ fontSize: '0.8rem', color: 'var(--accent-rose)', marginTop: '0.5rem' }}>⚠️ Key saved but connection failed — check the key</p>
                )}
              </div>
            </div>

            {/* Connection Diagnostics */}
            <div className="card">
              <h3 className="card-title">🔌 Connection Diagnostics</h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem 0', borderBottom: '1px solid var(--glass-border)' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>FastAPI Gateway:</span>
                  <span style={{ color: backendConnected ? 'var(--accent-emerald)' : 'var(--accent-rose)', fontWeight: 600 }}>
                    {backendConnected ? 'Operational (Port 8000)' : 'Offline / Unreachable'}
                  </span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem 0', borderBottom: '1px solid var(--glass-border)' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Groq Cloud:</span>
                  <span style={{ color: groqConnected ? 'var(--accent-emerald)' : 'var(--accent-rose)', fontWeight: 600 }}>
                    {groqConnected ? 'Connected (~300 tok/s)' : groqStatus}
                  </span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem 0', borderBottom: '1px solid var(--glass-border)' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Ollama Daemon (Local):</span>
                  <span style={{ color: ollamaConnected ? 'var(--accent-emerald)' : 'var(--accent-rose)', fontWeight: 600 }}>
                    {ollamaConnected ? 'Connected' : 'Offline'}
                  </span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.75rem 0', borderBottom: '1px solid var(--glass-border)' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Local Model Pool:</span>
                  <span>{models.join(', ') || 'No models found'}</span>
                </div>
              </div>

              <button 
                className="btn-secondary" 
                style={{ marginTop: '1.5rem', width: '100%' }}
                onClick={() => {
                  checkHealth();
                  fetchModels();
                  fetchGroqModels();
                }}
              >
                Refresh All Connections
              </button>
            </div>
            
            <div className="card">
              <h3 className="card-title">🚀 Local Model Setup (Optional)</h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.6, marginBottom: '1rem' }}>
                For offline usage, you can install local models via Ollama. Cloud mode (Groq) is recommended for speed.
              </p>
              <div style={{ background: 'rgba(15, 23, 42, 0.4)', padding: '1rem', borderRadius: '8px', borderLeft: '3px solid var(--accent-primary)', fontSize: '0.85rem', fontFamily: 'monospace', lineHeight: 1.6 }}>
                # 1. Qwen-2 Math (Recommended for local)<br />
                ollama pull qwen2-math<br /><br />
                # 2. DeepSeek-R1 (Advanced reasoning)<br />
                ollama pull deepseek-r1<br /><br />
                # 3. Llama 3 (General purpose)<br />
                ollama pull llama3
              </div>
            </div>
          </div>
        )}

        {/* --- HIGH-FIDELITY PRINT-ONLY WORKSHEET VIEW --- */}
        {currentQuestions.length > 0 && (
          <div className="print-only-layout">
            <div className="print-header">
              <h1>Mathematics Practice Worksheet</h1>
              <div className="print-subheader">
                <span><strong>Cognitive Tier:</strong> {selectedBloom} Level</span>
                <span><strong>Source File:</strong> {file?.name || 'Workspace Document'}</span>
                <span><strong>Date:</strong> {new Date().toLocaleDateString()}</span>
              </div>
            </div>
            
            <div className="print-questions-section">
              <h2>Problems</h2>
              <div className="print-divider"></div>
              {currentQuestions.map((q, idx) => (
                <div key={q.id || idx} className="print-question-card">
                  <div className="print-question-title">
                    Question {idx + 1} — <span className="print-topic">{q.topic || 'Math Problem'}</span>

                  </div>
                  <div className="print-question-body">
                    <MathRenderer text={q.question || ''} />
                  </div>
                  {/* Space for Student to write details */}
                  <div className="print-student-space">
                    <p style={{ fontSize: '0.8rem', color: '#888', fontStyle: 'italic', margin: 0 }}>Show working here:</p>
                    <div style={{ height: '110px' }}></div>
                  </div>
                </div>
              ))}
            </div>

            {includeSolutions && (
              <div className="print-solutions-section">
                <div className="page-break"></div>
                <h1>Practice Answer Key</h1>
                <div className="print-subheader">
                  <span><strong>Assessment Tier:</strong> {selectedBloom}</span>
                  <span><strong>Target Concept:</strong> {file?.name || 'Workspace Document'}</span>
                </div>
                <div className="print-divider"></div>
                
                <div className="print-solutions-list">
                  {currentQuestions.map((q, idx) => {
                    const steps = Array.isArray(q.solution_steps) 
                      ? q.solution_steps 
                      : (typeof q.solution_steps === 'string' ? [q.solution_steps] : []);
                    return (
                      <div key={q.id || idx} className="print-solution-card">
                        <div className="print-question-title" style={{ color: '#000' }}>
                          Solution Key {idx + 1} — {q.topic || 'Math Problem'}
                        </div>
                        <div className="print-question-body" style={{ background: '#f5f5f5', border: '1px solid #ddd', padding: '10px', borderRadius: '4px', margin: '5px 0' }}>
                          <MathRenderer text={q.question || ''} />
                        </div>
                        <div style={{ fontSize: '0.95rem', margin: '8px 0' }}>
                          <strong>Correct Answer:</strong> <MathRenderer text={q.answer || ''} />
                        </div>
                        <div style={{ fontSize: '0.9rem', color: '#444' }}>
                          <strong>Step-by-Step Derivation:</strong>
                          <ol style={{ paddingLeft: '20px', marginTop: '5px' }}>
                            {steps.map((step, sIdx) => (
                              <li key={sIdx} style={{ marginBottom: '4px' }}>
                                <MathRenderer text={step || ''} />
                              </li>
                            ))}
                          </ol>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

      </main>
    </div>
  );
}
