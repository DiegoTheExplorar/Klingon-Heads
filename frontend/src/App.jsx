import React from 'react';
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import HistoryPage from './HistoryPage';
import SignIn from './SignIn';
import Translator from './Translator';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SignIn />} />
        <Route path="/translator" element={<Translator />} />
        <Route path="/history" element={<HistoryPage />} />
      </Routes>
    </Router>
  );
};

export default App;
