import React from 'react';
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import LandingPage from './LandingPage';
import Translator from './Translator';

const App = () => {
  return (
    <Router>
      <div>
        <Routes> {/* Updated from Switch to Routes */}
          <Route exact path="/" element={<LandingPage />} /> {/* Updated component to element */}
          <Route path="/translator" element={<Translator />} /> {/* Updated component to element and path case */}
        </Routes>
      </div>
    </Router>
  );
};

export default App;
