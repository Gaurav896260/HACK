import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import {
  FaSearch,
  FaBuilding,
  FaRegLightbulb,
  FaLightbulb,
  FaMagic,
  FaQuestionCircle,
  FaThumbsUp,
  FaThumbsDown,
  FaExclamationTriangle,
  FaInfoCircle,
  FaRobot,
  FaHistory,
  FaMicrophone,
  FaMicrophoneSlash,
  FaCamera,
  FaImage,
  FaSpinner,
} from "react-icons/fa";

function App() {
  const [businessDesc, setBusinessDesc] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [originalInput, setOriginalInput] = useState("");
  const [grammarSuggestion, setGrammarSuggestion] = useState("");
  const [vagueSuggestions, setVagueSuggestions] = useState([]);
  const [isVague, setIsVague] = useState(false);
  const [searchHistory, setSearchHistory] = useState([]);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState({});
  const [showInfoModal, setShowInfoModal] = useState(false);
  const [activeTab, setActiveTab] = useState("results");
  const [enhancedInput, setEnhancedInput] = useState("");
  const [showEnhancedModal, setShowEnhancedModal] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [imagePredictions, setImagePredictions] = useState([]);
  const [showImageModal, setShowImageModal] = useState(false);
  const recognitionRef = useRef(null);
  const fileInputRef = useRef(null);

  // Load search history from localStorage on component mount
  useEffect(() => {
    const savedHistory = localStorage.getItem("searchHistory");
    if (savedHistory) {
      try {
        setSearchHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error("Error loading search history:", e);
      }
    }

    // Initialize speech recognition
    if ("SpeechRecognition" in window || "webkitSpeechRecognition" in window) {
      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;

      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map((result) => result[0])
          .map((result) => result.transcript)
          .join("");

        setBusinessDesc(transcript);
      };

      recognitionRef.current.onerror = (event) => {
        console.error("Speech recognition error", event.error);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        if (isListening) {
          recognitionRef.current.start();
        }
      };
    }

    return () => {
      // Clean up speech recognition on component unmount
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  // Save search history to localStorage when it changes
  useEffect(() => {
    if (searchHistory.length > 0) {
      localStorage.setItem("searchHistory", JSON.stringify(searchHistory));
    }
  }, [searchHistory]);

  // Toggle speech recognition
  const toggleListening = () => {
    if (!recognitionRef.current) {
      alert("Speech recognition is not supported in your browser.");
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.start();
      setIsListening(true);
    }
  };

  // Handle image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSelectedImage(file);

    // Create image preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
      setShowImageModal(true);
    };
    reader.readAsDataURL(file);
  };

  // Trigger file input click
  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  // Process uploaded image
  const processImage = async () => {
    if (!selectedImage) return;

    setImageLoading(true);
    try {
      const formData = new FormData();
      formData.append("image", selectedImage);

      const response = await fetch("http://localhost:5000/analyze_image", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to analyze image");
      }

      const data = await response.json();
      setImagePredictions(data.predictions || []);
      setBusinessDesc(data.suggested_description || "");
      setShowImageModal(false);
    } catch (error) {
      console.error("Error processing image:", error);
      alert("Failed to analyze image. Please try again.");
    } finally {
      setImageLoading(false);
    }
  };

  const enhanceBusinessDescription = async (rawInput) => {
    if (!rawInput.trim()) return "";

    try {
      const prompt = `
You are an AI assistant specialized in interpreting business descriptions and mapping them to standardized industry classifications. Your task is to analyze the given business description, understand the underlying context and activities, and rewrite it in a clear, structured format suitable for classification into NIC (National Industrial Classification) codes.

Guidelines:

Interpret vague or general descriptions, inferring the most likely specific activities.

Use precise, industry-standard terminology where appropriate.

Break down complex businesses into their core components or activities.

Provide enough detail to distinguish between similar but distinct categories.

Focus on the primary business activities, not ancillary operations.

Maintain accuracy while making the description more specific and classifiable.

Rewrite the following business description in a clear, structured format optimized for NIC code classification. Return ONLY the refined business description without any additional explanations or commentary.

Business description: "${rawInput}"
`;
      console.log("Prompt for Gemini API:", prompt);
      const response = await fetch(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-goog-api-key": "AIzaSyC1GybTjiD6wyRKK7beet_ZY-MCsN9GAvo", // Replace with your actual API key
          },
          body: JSON.stringify({
            contents: [
              {
                parts: [
                  {
                    text: prompt,
                  },
                ],
              },
            ],
            generationConfig: {
              temperature: 0.2, // Lower temperature for more focused output
              maxOutputTokens: 100, // Limit output length
            },
          }),
        }
      );

      const data = await response.json();

      if (data.candidates && data.candidates[0] && data.candidates[0].content) {
        const enhancedDescription =
          data.candidates[0].content.parts[0].text.trim();
        console.log("Enhanced description:", enhancedDescription);
        return enhancedDescription;
      } else {
        console.error("Unexpected Gemini API response format:", data);
        return rawInput; // Fall back to original input
      }
    } catch (error) {
      console.error("Error enhancing business description:", error);
      return rawInput; // Fall back to original input on error
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // If still listening, stop speech recognition
    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }

    const query = businessDesc.trim();
    if (!query) {
      alert("Please describe your business activity");
      return;
    }

    setLoading(true);
    setShowResults(false);
    setIsVague(false);
    setVagueSuggestions([]);
    setFeedbackSubmitted({});

    try {
      // First, enhance the business description using Gemini API
      const enhancedQuery = await enhanceBusinessDescription(query);

      // Store both the original and enhanced queries
      const originalQuery = query;

      // Display a notice if the description was enhanced
      if (enhancedQuery !== query && enhancedQuery.trim() !== "") {
        setGrammarSuggestion(enhancedQuery);
      }

      // Use the enhanced query (or fall back to original) for the NIC code search
      const queryToUse = enhancedQuery.trim() !== "" ? enhancedQuery : query;

      const response = await fetch("http://localhost:5000/get_nic_codes", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input: queryToUse,
          original_input: originalQuery,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch results");
      }

      const data = await response.json();
      setResults(data.results);
      setSuggestions(data.suggestions || []);
      setOriginalInput(data.original_input);

      // Handle grammar suggestions (if not already set by Gemini)
      if (
        !grammarSuggestion &&
        data.corrected_input &&
        data.corrected_input !== queryToUse
      ) {
        setGrammarSuggestion(data.corrected_input);
      }

      // Handle vague query suggestions
      if (data.is_vague) {
        setIsVague(true);
        setVagueSuggestions(data.vague_suggestions || []);
      }

      // Add to search history if we got results
      if (data.results.length > 0) {
        const newSearch = {
          query: query,
          timestamp: new Date().toISOString(),
          topResult: data.results[0]?.nic_code,
        };

        // Add to history, keeping only the most recent 10 searches
        setSearchHistory((prev) => {
          const updated = [
            newSearch,
            ...prev.filter((item) => item.query !== query),
          ].slice(0, 10);
          return updated;
        });
      }

      setShowResults(true);
      setActiveTab("results");
    } catch (error) {
      console.error("Error:", error);
      alert("Something went wrong. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setBusinessDesc(suggestion);
  };

  const applyGrammarCorrection = () => {
    setBusinessDesc(grammarSuggestion);
    setGrammarSuggestion("");
  };

  const handleVagueSuggestionClick = (suggestion) => {
    setBusinessDesc((prev) => {
      if (prev.trim() === "") {
        return suggestion;
      } else {
        return `${prev} (${suggestion})`;
      }
    });
  };

  const handleHistoryItemClick = (historyItem) => {
    setBusinessDesc(historyItem.query);
  };

  const submitFeedback = async (resultId, isPositive) => {
    // Prevent multiple feedback submissions for the same result
    if (feedbackSubmitted[resultId] !== undefined) {
      return;
    }

    try {
      await fetch("http://localhost:5000/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          original_query: originalInput,
          nic_code: results[resultId].nic_code,
          is_positive: isPositive,
        }),
      });

      // Update feedback state
      setFeedbackSubmitted((prev) => ({
        ...prev,
        [resultId]: isPositive,
      }));
    } catch (error) {
      console.error("Error submitting feedback:", error);
    }
  };

  const clearHistory = () => {
    setSearchHistory([]);
    localStorage.removeItem("searchHistory");
  };

  return (
    <div className="app-container">
      {showInfoModal && (
        <div className="modal-backdrop">
          <div className="info-modal">
            <h2>
              About NIC Code Finder <FaInfoCircle />
            </h2>
            <p>
              The National Industrial Classification (NIC) code is a statistical
              standard for organizing economic data. This tool helps you find
              the appropriate NIC code for your business activities.
            </p>
            <h3>Tips for better results:</h3>
            <ul>
              <li>Be specific about your business activities</li>
              <li>Include key products or services you provide</li>
              <li>Mention manufacturing processes or methods if applicable</li>
              <li>Use industry-specific terminology when possible</li>
              <li>
                You can now use the microphone icon to dictate your business
                description
              </li>
              <li>
                Try the new image upload feature to analyze products in your
                business
              </li>
            </ul>
            <button onClick={() => setShowInfoModal(false)}>Close</button>
          </div>
        </div>
      )}

      {showImageModal && (
        <div className="modal-backdrop">
          <div className="image-analysis-modal">
            <h2>
              <FaImage /> Image Analysis
            </h2>
            <div className="image-preview-container">
              {imagePreview && (
                <img
                  src={imagePreview}
                  alt="Uploaded business"
                  className="image-preview"
                />
              )}
            </div>
            <p>
              Upload a photo of your business or products to help identify the
              appropriate NIC code.
            </p>
            {imageLoading ? (
              <div className="image-processing-indicator">
                <FaSpinner className="spinning-icon" />
                <p>Analyzing image...</p>
              </div>
            ) : (
              <div className="image-modal-buttons">
                <button onClick={() => setShowImageModal(false)}>Cancel</button>
                <button onClick={processImage} className="process-image-button">
                  Analyze Image
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="content-wrapper">
        <header>
          <div className="header-content">
            <h1>
              <FaBuilding className="icon" /> NIC Code Finder
            </h1>
            <button
              className="info-button"
              onClick={() => setShowInfoModal(true)}
            >
              <FaQuestionCircle />
            </button>
          </div>
          <p>
            Find the perfect National Industrial Classification code for your
            business activities
          </p>
        </header>

        <main>
          <div className="search-container">
            {grammarSuggestion && (
              <div className="grammar-suggestion">
                <FaMagic className="magic-icon" />
                <span>
                  <strong>Did you mean:</strong> {grammarSuggestion}
                </span>
                <button onClick={applyGrammarCorrection}>Apply</button>
              </div>
            )}

            {imagePredictions.length > 0 && (
              <div className="image-predictions-container">
                <h4>
                  <FaCamera /> Image Analysis Results
                </h4>
                <div className="predictions-list">
                  {imagePredictions.map((prediction, index) => (
                    <div key={index} className="prediction-item">
                      <span className="prediction-label">
                        {prediction.label}
                      </span>
                      <span className="prediction-probability">
                        {(prediction.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {enhancedInput && enhancedInput !== businessDesc && (
              <div className="enhanced-query-notice">
                <FaRobot className="ai-icon" />
                <span>
                  <strong>AI-Enhanced:</strong> Your query was processed with
                  Gemini AI to improve results
                </span>
                <button
                  className="view-enhanced-button"
                  onClick={() => setShowEnhancedModal(true)}
                >
                  View Enhanced Query
                </button>
              </div>
            )}
            {showEnhancedModal && (
              <div className="modal-backdrop">
                <div className="info-modal">
                  <h2>
                    <FaRobot /> AI-Enhanced Query
                  </h2>
                  <div className="query-comparison">
                    <div className="original-query">
                      <h3>Your Original Description:</h3>
                      <p>{originalInput}</p>
                    </div>
                    <div className="enhanced-query">
                      <h3>AI-Enhanced Description:</h3>
                      <p>{enhancedInput}</p>
                    </div>
                  </div>
                  <p className="enhancement-explanation">
                    Gemini AI analyzed your business description and extracted
                    the most relevant industry terms and activities to improve
                    your NIC code matching results.
                  </p>
                  <button onClick={() => setShowEnhancedModal(false)}>
                    Close
                  </button>
                </div>
              </div>
            )}
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="businessDesc">
                  <FaRegLightbulb className="input-icon" />
                  Describe your business activity
                </label>
                <div className="input-controls">
                  <div className="textarea-with-mic">
                    <textarea
                      id="businessDesc"
                      value={businessDesc}
                      onChange={(e) => setBusinessDesc(e.target.value)}
                      placeholder="e.g. Manufacture of mineral water, Production of cement, Software development services"
                      rows="3"
                    />
                    <div className="input-buttons">
                      <button
                        type="button"
                        className={`mic-button ${
                          isListening ? "listening" : ""
                        }`}
                        onClick={toggleListening}
                        title={
                          isListening ? "Stop listening" : "Start voice input"
                        }
                      >
                        {isListening ? (
                          <FaMicrophone className="mic-icon active" />
                        ) : (
                          <FaMicrophoneSlash className="mic-icon" />
                        )}
                      </button>
                      <button
                        type="button"
                        className="camera-button"
                        onClick={triggerFileInput}
                        title="Upload business image"
                      >
                        <FaCamera className="camera-icon" />
                      </button>
                      <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleImageUpload}
                        accept="image/*"
                        style={{ display: "none" }}
                      />
                    </div>
                  </div>
                  {isListening && (
                    <div className="listening-indicator">
                      <span className="pulse-dot"></span> Listening... speak now
                    </div>
                  )}
                </div>
              </div>

              {isVague && vagueSuggestions.length > 0 && (
                <div className="vague-query-container">
                  <h4>
                    <FaExclamationTriangle className="warning-icon" /> Your
                    description is a bit general
                  </h4>
                  <p>Try adding more details or select a specific industry:</p>
                  <div className="vague-suggestions">
                    {vagueSuggestions.map((suggestion, index) => (
                      <button
                        key={index}
                        className="vague-suggestion-button"
                        onClick={() => handleVagueSuggestionClick(suggestion)}
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {suggestions.length > 0 && (
                <div className="suggestions-container">
                  <h4>
                    <FaLightbulb /> Expanded Search Terms
                  </h4>
                  <div className="suggestions-chips">
                    {suggestions.map((suggestion, index) => (
                      <span
                        key={index}
                        className="suggestion-chip"
                        onClick={() => handleSuggestionClick(suggestion)}
                      >
                        {suggestion}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <button type="submit" className="search-button">
                <FaSearch className="button-icon" />
                Find NIC Codes
              </button>
            </form>
          </div>

          {loading && (
            <div className="loading-container">
              <div className="loader"></div>
              <p>Analyzing your business description...</p>
              <p className="loading-subtext">
                <FaRobot className="ai-icon" /> AI is processing
                industry-specific terms and patterns
              </p>
            </div>
          )}

          {showResults && (
            <div className="results-section">
              <div className="results-tabs">
                <button
                  className={`tab-button ${
                    activeTab === "results" ? "active" : ""
                  }`}
                  onClick={() => setActiveTab("results")}
                >
                  Results
                </button>
                <button
                  className={`tab-button ${
                    activeTab === "history" ? "active" : ""
                  }`}
                  onClick={() => setActiveTab("history")}
                >
                  <FaHistory /> Search History
                </button>
              </div>

              {activeTab === "results" && (
                <div className="results-container">
                  <h2>Top NIC Code Suggestions:</h2>
                  {results.length === 0 ? (
                    <div className="no-results">
                      <FaExclamationTriangle className="warning-icon" />
                      <p>
                        No matching NIC codes found. Try adding more specific
                        details about your business activities.
                      </p>
                    </div>
                  ) : (
                    <div className="results-cards">
                      {results.map((result, index) => (
                        <div className="result-card" key={index}>
                          <div className="card-header">
                            <h3>NIC Code: {result.nic_code}</h3>
                            <span
                              className={`match-badge ${
                                result.similarity_score > 0.7
                                  ? "high-match"
                                  : result.similarity_score > 0.4
                                  ? "medium-match"
                                  : "low-match"
                              }`}
                            >
                              Match {(result.similarity_score * 100).toFixed(1)}
                              %
                            </span>
                          </div>
                          <p className="result-description">
                            {result.description}
                          </p>
                          <div className="result-details">
                            <p>
                              <strong>Division:</strong> {result.division}
                            </p>
                            <p>
                              <strong>Section:</strong> {result.section}
                            </p>
                          </div>
                          <div className="feedback-buttons">
                            <p>Was this helpful?</p>
                            <div>
                              <button
                                className={`feedback-button ${
                                  feedbackSubmitted[index] === true
                                    ? "selected"
                                    : ""
                                }`}
                                onClick={() => submitFeedback(index, true)}
                                disabled={
                                  feedbackSubmitted[index] !== undefined
                                }
                              >
                                <FaThumbsUp />
                              </button>
                              <button
                                className={`feedback-button ${
                                  feedbackSubmitted[index] === false
                                    ? "selected"
                                    : ""
                                }`}
                                onClick={() => submitFeedback(index, false)}
                                disabled={
                                  feedbackSubmitted[index] !== undefined
                                }
                              >
                                <FaThumbsDown />
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {activeTab === "history" && (
                <div className="history-container">
                  <div className="history-header">
                    <h2>Your Recent Searches</h2>
                    {searchHistory.length > 0 && (
                      <button
                        className="clear-history-button"
                        onClick={clearHistory}
                      >
                        Clear History
                      </button>
                    )}
                  </div>
                  {searchHistory.length === 0 ? (
                    <p>No search history yet.</p>
                  ) : (
                    <div className="history-items">
                      {searchHistory.map((item, index) => (
                        <div
                          key={index}
                          className="history-item"
                          onClick={() => handleHistoryItemClick(item)}
                        >
                          <p className="history-query">{item.query}</p>
                          <div className="history-details">
                            <span className="history-date">
                              {new Date(item.timestamp).toLocaleDateString()}
                            </span>
                            <span className="history-code">
                              NIC: {item.topResult}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </main>

        <footer>
          <p>© 2025 NIC Code Finder | Powered by AI Technology</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
