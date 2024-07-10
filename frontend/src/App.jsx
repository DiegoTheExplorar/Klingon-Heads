import { getAuth, onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { Navigate, Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import FavoritesPage from './History_and_Favs/FavoritesPage';
import HistoryPage from './History_and_Favs/HistoryPage';
import LandingPage from './LandingPage';
import FetchDataComponent from './Learn/ShowCards';
import MainLayout from './MainLayout';
import EnglishQuiz from './Quiz/EnglishQuiz';
import KlingonQuiz from './Quiz/KlingonQuiz';
import RandomQuiz from './Quiz/RandomQuiz';
import StartQuiz from './Quiz/StartQuiz';
import SignIn from './SignIn';
import Translator from './Translator';

const PrivateRoute = ({ element, isLoggedIn }) => {
  return isLoggedIn ? element : <Navigate to="/signin" />;
};

const App = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const auth = getAuth();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, user => {
      setIsLoggedIn(!!user);
    });

    return () => unsubscribe();
  }, [auth]);

  const routes = [
    { path: "/", element: <LandingPage /> },
    { path: "/signin", element: <SignIn /> },
    { path: "/translator", element: <Translator />, protected: true },
    { path: "/fav", element: <FavoritesPage />, protected: true },
    { path: "/history", element: <HistoryPage />, protected: true },
    { path: "/learn", element: <FetchDataComponent />, protected: true },
    { path: "/quiz", element: <StartQuiz />, protected: true },
    { path: "/english-quiz", element: <EnglishQuiz />, protected: true },
    { path: "/klingon-quiz", element: <KlingonQuiz /> , protected: true},
    { path: "/random-quiz", element: <RandomQuiz />, protected: true },
  ];

  return (
    <Router>
      <Routes>
        {routes.map(({ path, element, protected: isProtected }) => (
          <Route
            key={path}
            path={path}
            element={
              isProtected ? (
                <PrivateRoute element={<MainLayout>{element}</MainLayout>} isLoggedIn={isLoggedIn} />
              ) : (
                element
              )
            }
          />
        ))}
      </Routes>
    </Router>
  );
};

export default App;
