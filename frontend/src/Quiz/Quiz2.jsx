import { getAuth, onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import './Quiz2.css';
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
  const [timer, setTimer] = useState(30);
  const [selectedOption, setSelectedOption] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const auth = getAuth();

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
    setTimer(30);
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
    setIsSubmitted(true);
  };

  const handleNextQuestion = () => {
    const nextQuestionIndex = currentQuestionIndex + 1;
    if (nextQuestionIndex < questions.length) {
      setCurrentQuestionIndex(nextQuestionIndex);
      setSelectedOption('');
      setIsSubmitted(false);
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

  const handleOptionSelect = option => {
    if (!isSubmitted) {
      setSelectedOption(option);
    }
  };

  const handleSubmit = () => {
    if (!isSubmitted) {
      onAnswerSubmit(selectedOption === questions[currentQuestionIndex].options[questions[currentQuestionIndex].correct_index]);
    } else {
      handleNextQuestion();
    }
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
          <div className="question-container">
            <div className="question-box">
              <h3>{questions[currentQuestionIndex].question}</h3>
              {questions[currentQuestionIndex].options.map((option, index) => (
                <div key={index} className={`option ${selectedOption === option ? 'selected' : ''} ${isSubmitted && (index === questions[currentQuestionIndex].correct_index ? 'correct' : (selectedOption === option ? 'incorrect' : ''))}`}>
                  <button onClick={() => handleOptionSelect(option)} disabled={isSubmitted} className={selectedOption === option ? 'selected' : ''}>
                    <span className="option-label">{String.fromCharCode(65 + index)}</span> {option}
                  </button>
                </div>
              ))}
              <button onClick={handleSubmit} className="submit-button">
                {isSubmitted ? (currentQuestionIndex + 1 === questions.length ? 'See Summary' : 'Next Question') : 'Submit'}
              </button>
            </div>
          </div>
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
