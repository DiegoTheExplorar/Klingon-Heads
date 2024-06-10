import React from 'react';
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import HistoryPage from './HistoryPage'; // Import the HistoryPage
import LandingPage from './LandingPage';
import Translator from './Translator';

const App = () => {
  return (
    <Router>
      <div>
        <Routes>
          <Route exact path="/" element={<LandingPage />} />
          <Route path="/translator" element={<Translator />} />
          <Route path="/history" element={<HistoryPage />} /> {/* Route for the History Page */}
        </Routes>
      </div>
    </Router>
  );
};

export default App;
