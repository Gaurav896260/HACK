import React, { useState } from "react";
import "./App.css";
import { FaSearch, FaBuilding, FaRegLightbulb } from "react-icons/fa";

function App() {
  const [businessDesc, setBusinessDesc] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!businessDesc.trim()) {
      alert("Please describe your business activity");
      return;
    }

    setLoading(true);
    setShowResults(false);

    try {
      const response = await fetch("http://localhost:5000/get_nic_codes", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input: businessDesc }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch results");
      }

      const data = await response.json();
      setResults(data);
      setShowResults(true);
    } catch (error) {
      console.error("Error:", error);
      alert("Something went wrong. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="content-wrapper">
        <header>
          <h1>
            <FaBuilding className="icon" /> NIC Code Finder
          </h1>
          <p>
            Find the perfect National Industrial Classification code for your
            business
          </p>
        </header>

        <main>
          <div className="search-container">
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="businessDesc">
                  <FaRegLightbulb className="input-icon" />
                  Describe your business activity
                </label>
                <textarea
                  id="businessDesc"
                  value={businessDesc}
                  onChange={(e) => setBusinessDesc(e.target.value)}
                  placeholder="e.g. Manufacture of mineral water, Production of cement, Software development services"
                  rows="3"
                />
              </div>

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
            </div>
          )}

          {showResults && (
            <div className="results-container">
              <h2>Top NIC Code Suggestions:</h2>
              <div className="results-cards">
                {results.map((result, index) => (
                  <div className="result-card" key={index}>
                    <div className="card-header">
                      <h3>NIC Code: {result.nic_code}</h3>
                      <span className="match-badge">Match {index + 1}</span>
                    </div>
                    <p>{result.description}</p>
                  </div>
                ))}
              </div>
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
