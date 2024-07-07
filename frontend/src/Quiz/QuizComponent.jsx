import { getAuth, onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import UserDropdown from '../UserDropdown';
import './QuizComponent.css';
import QuizQuestion from './QuizQuestion';
import QuizSummary from './QuizSummary';
import StartQuiz from './StartQuiz';

function QuizComponent() {
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [finished, setFinished] = useState(false);
  const [started, setStarted] = useState(false);
  const [submittedCount, setSubmittedCount] = useState(0);
  const [showDropdown, setShowDropdown] = useState(false);
  const [profilePicUrl, setProfilePicUrl] = useState(null);
  const [timer, setTimer] = useState(5);
  const auth = getAuth();
  const [Wrong, setWrong] = useState([]);
  useEffect(() => {
    fetchQuestions();
    const timerId = started && !finished && setInterval(() => {
      setTimer(prevTimer => (prevTimer > 0 ? prevTimer - 1 : 0));
    }, 1000);
    return () => clearInterval(timerId);
  }, [started, finished]);

  useEffect(() => {
    if (timer === 0 && started && !finished) {
      handleAnswerSubmit(false);
    }
  }, [timer, started, finished]);

  useEffect(() => {
    setTimer(5);
  }, [currentQuestionIndex]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, user => {
      if (user) {
        setProfilePicUrl(user.photoURL);
      } else {
        setProfilePicUrl(null);
      }
    });
    return () => unsubscribe();
  }, [auth]);

  const fetchQuestions = async () => {
    try {
      const response = await fetch('https://klingonapi-cafaedb94044.herokuapp.com/quiz');
      const data = await response.json();
      setQuestions(data);
    } catch (error) {
      console.error('Error fetching questions:', error);
    }
  };

  const handleAnswerSubmit = isCorrect => {
    if (isCorrect) {
      setScore(prevScore => prevScore + 1 + timer);
    }
    setSubmittedCount(submittedCount + 1);
  };

  const handleNextQuestion = () => {
    const nextQuestionIndex = currentQuestionIndex + 1;
    if (nextQuestionIndex < questions.length) {
      setCurrentQuestionIndex(nextQuestionIndex);
    } else {
      setFinished(true);
    }
  };

  

  const handleRestartQuiz = () => {
    setCurrentQuestionIndex(0);
    setScore(0);
    setFinished(false);
    setStarted(true);  
    setSubmittedCount(0);
    fetchQuestions();
  };

  return (
    <div>
      {!started ? (
        <StartQuiz onStartQuiz={() => setStarted(true)} />
      ) : !finished ? (
        <div className="quiz-container">
          <div className="progress-bar">
            <div className="progress" style={{ width: `${(submittedCount / questions.length) * 100}%` }}></div>
          </div>
          <div className="score-tracker">
            Score: {score} / {questions.length}
          </div>
          <div className="timer">
            Time Remaining: {timer}s
          </div>
          <QuizQuestion
          question={questions[currentQuestionIndex].question}
          options={questions[currentQuestionIndex].options}
          correctIndex={questions[currentQuestionIndex].correct_index}
          onAnswerSubmit={handleAnswerSubmit}
          onNextQuestion={handleNextQuestion}
          currentNumber={currentQuestionIndex}
          totalQuestions={questions.length}
        />
        </div>
      ) : (
        <QuizSummary score={score} totalQuestions={questions.length} onRestartQuiz={handleRestartQuiz} />
      )}
      <div className="user-icon-container" onClick={() => setShowDropdown(!showDropdown)}>
        {profilePicUrl ? (
          <img src={profilePicUrl} alt="User Icon" className="user-profile-pic" />
        ) : (
          <div className="user-icon" />
        )}
        {showDropdown && <UserDropdown auth={auth} profilePicUrl={profilePicUrl} />}
      </div>
    </div>
  );
}

export default QuizComponent;