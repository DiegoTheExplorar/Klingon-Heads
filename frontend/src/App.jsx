import React from 'react';
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import Auth from './Auth';
import HistoryPage from './HistoryPage';
import LandingPage from './LandingPage';
import PrivateRoute from './PrivateRoute';
import Translator from './Translator';

const App = () => {
  return (
    <Router>
      <div>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/translator" element={
            <PrivateRoute>
              <Translator />
            </PrivateRoute>
          } />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/auth" element={<Auth />} /> {/* Route for Authentication */}
        </Routes>
      </div>
    </Router>
  );
};

export default App;
