import { getAuth, onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { Navigate, Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import FavoritesPage from './FavoritesPage';
import HistoryPage from './HistoryPage';
import LandingPage from './LandingPage';
import FetchDataComponent from './Learn/ShowCards';
import MatchingQuiz from './Quiz/MatchingQuiz';
import SignIn from './SignIn';
import Translator from './Translator';

const quizData = [
  { id: 1, english: "Hello", klingon: "nuqneH" },
  { id: 2, english: "Goodbye", klingon: "Qapla'" },
  { id: 3, english: "Thank you", klingon: "tlho'" },
  { id: 4, english: "Yes", klingon: "HIja'" },
  { id: 5, english: "No", klingon: "ghobe'" }
];
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
        <Route path="/quiz" element={<MatchingQuiz quizData={quizData}/>} />
      </Routes>
    </Router>
  );
};

export default App;
