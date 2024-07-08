import { getAuth, onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import UserDropdown from '../UserDropdown';
import './QuizComponent.css';
import QuizQuestion from './QuizQuestion';
import QuizSummary from './QuizSummary';
import StartQuiz from './StartQuiz';
const time = 5;

const TimeUpModal = ({ onClose }) => (
  <div className="modal-backdrop">
    <div className="modal-content">
      <h4>Time's Up!</h4>
      <p>You didn't answer in time. Please try to be quicker next time!</p>
      <button onClick={onClose} style={{ alignSelf: 'center' }}>Close</button>
    </div>
  </div>
);

function QuizComponent() {
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [finished, setFinished] = useState(false);
  const [started, setStarted] = useState(false);
  const [submittedCount, setSubmittedCount] = useState(0);
  const [showDropdown, setShowDropdown] = useState(false);
  const [profilePicUrl, setProfilePicUrl] = useState(null);
  const [timer, setTimer] = useState(time);
  const auth = getAuth();
  const [Wrong,addWrong] = useState([]);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    fetchQuestions();
  }, []);

  useEffect(() => {
    let timerId;
    if (started && !finished && timer > 0) {  
      timerId = setInterval(() => {
        setTimer(prevTimer => (prevTimer > 0 ? prevTimer - 1 : 0));
      }, 1000);
    }
    return () => clearInterval(timerId);
  }, [started, finished, timer]);
  

  useEffect(() => {
    if (timer === 0 && started && !finished) {
      setShowModal(true);
      setTimeout(() => {
        setShowModal(false);
        handleAnswerSubmit(false); 
      }, 3000); 
    }
  }, [timer, started, finished]);
  

  useEffect(() => {
    setTimer(time);
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
    setTimer(-1)
    const currentQuestion = questions[currentQuestionIndex];
    if (!isCorrect) {
      addWrong([...Wrong, {
        question: currentQuestion.question,
        correctOption: currentQuestion.options[currentQuestion.correct_index]
      }]);
    }
    if (isCorrect) {
      setScore(prevScore => prevScore + timer);
    }
    if(currentQuestionIndex === questions.length - 1) {
      setSubmittedCount(prevCount => prevCount + 1);
    }
  };
  
  

  const handleNextQuestion = () => {
    setSubmittedCount(submittedCount + 1);
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
    addWrong([]);
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
            Score: {score}
          </div>
          <div className="timer">
          Time Remaining: {timer === -1 ? 0 : timer}s
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
        {showModal && <TimeUpModal onClose={() => setShowModal(false)} />}
        </div>
      ) : (
        <QuizSummary score={score} onRestartQuiz={handleRestartQuiz} Wrong = {Wrong}/>
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