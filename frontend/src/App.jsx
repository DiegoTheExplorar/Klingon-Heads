import { getAuth, onAuthStateChanged } from 'firebase/auth'; // Ensure to import `onAuthStateChanged`
import React, { useEffect, useState } from 'react';
import { Navigate, Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import FavoritesComponent from './FavoritesComponent';
import HistoryPage from './HistoryPage';
import SignIn from './SignIn';
import Translator from './Translator';

const App = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const auth = getAuth(); // Initialize Firebase Auth

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, user => {
      setIsLoggedIn(!!user);
    });

    return () => unsubscribe(); // Cleanup on unmount
  }, [auth]);

  const PrivateRoute = ({ children }) => {
    return isLoggedIn ? children : <Navigate to="/" />;
  };

  return (
    <Router>
      <Routes>
        <Route path="/" element={isLoggedIn ? <Navigate to="/translator" /> : <SignIn />} />
        <Route path="/translator" element={<Translator />} />

        <Route path="/history" element={
          <PrivateRoute>
            <HistoryPage />
          </PrivateRoute>
        } />

        <Route path="/fav" element={
          <PrivateRoute>
            <FavoritesComponent />
          </PrivateRoute>
        } />

        {/* Catch-all route */}
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </Router>
  );
};

export default App;
