import { getAuth, onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { Navigate, Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import FavoritesPage from './History_and_Favs/FavoritesPage';
import HistoryPage from './History_and_Favs/HistoryPage';
import LandingPage from './LandingPage';
import FetchDataComponent from './Learn/ShowCards';
import QuizComponent from './Quiz/Quiz2';
import SignIn from './SignIn';
import Translator from './Translator';

const App = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const auth = getAuth(); 

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, user => {
      setIsLoggedIn(!!user);
    });

    return () => unsubscribe(); 
  }, [auth]);

  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/translator" element={isLoggedIn ? <Translator /> : <Navigate to="/signin" />} />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/fav" element={isLoggedIn ? <FavoritesPage /> : <Navigate to="/signin" />} />
        <Route path="/history" element={isLoggedIn ? <HistoryPage /> : <Navigate to="/signin" />} />
        <Route path="/learn" element={<FetchDataComponent />} />
        <Route path="/quiz" element={<QuizComponent/>} />
      </Routes>
    </Router>
  );
};

export default App;
