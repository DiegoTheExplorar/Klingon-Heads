import React from 'react';
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import HistoryPage from './HistoryPage'; // Import the HistoryPage
import SignIn from './SignIn';
import Translator from './Translator';
const App = () => {
  return (
    <Router>
      <div>
        <Routes>
          <Route exact path="/" element={<SignIn />} /> {/* Render SignIn as default */}
          <Route path="/translator" element={<Translator />} />
          <Route path="/history" element={<HistoryPage />} /> {/* Route for the History Page */}
        </Routes>
      </div>
    </Router>
  );
};

export default App;
