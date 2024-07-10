import { getAuth, onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { Navigate, Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import FavoritesPage from './History_and_Favs/FavoritesPage';
import HistoryPage from './History_and_Favs/HistoryPage';
import LandingPage from './LandingPage';
import FetchDataComponent from './Learn/ShowCards';
import MainLayout from './MainLayout'; // Import the new layout
import EnglishQuiz from './Quiz/EnglishQuiz';
import KlingonQuiz from './Quiz/KlingonQuiz';
import RandomQuiz from './Quiz/RandomQuiz';
import StartQuiz from './Quiz/StartQuiz';
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
        <Route path="/signin" element={<SignIn />} />
        <Route path="/translator" element={isLoggedIn ? <MainLayout><Translator /></MainLayout> : <Navigate to="/signin" />} />
        <Route path="/fav" element={isLoggedIn ? <MainLayout><FavoritesPage /></MainLayout> : <Navigate to="/signin" />} />
        <Route path="/history" element={isLoggedIn ? <MainLayout><HistoryPage /></MainLayout> : <Navigate to="/signin" />} />
        <Route path="/learn" element={<MainLayout><FetchDataComponent /></MainLayout>} />
        <Route path="/quiz" element={<MainLayout><StartQuiz/></MainLayout>} />
        <Route path="/english-quiz" element={<MainLayout><EnglishQuiz /></MainLayout>} />
        <Route path="/klingon-quiz" element={<MainLayout><KlingonQuiz /></MainLayout>} />
        <Route path="/random-quiz" element={<MainLayout><RandomQuiz /></MainLayout>} />
      </Routes>
    </Router>
  );
};

export default App;
