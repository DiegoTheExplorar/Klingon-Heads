import React, { useEffect, useState } from 'react';
import './QuizComponent.css';
import QuizQuestion from './QuizQuestion';
import QuizSummary from './QuizSummary';
import { useNavigate } from 'react-router-dom';
import { Icon } from '@iconify/react';
import arrowBack from '@iconify-icons/mdi/arrow-back';

const TimeUpModal = ({ onClose }) => (
  <div className="modal-backdrop">
    <div className="modal-content">
      <h4>Time's Up!</h4>
      <p>You didn't answer in time. Please try to be quicker next time!</p>
      <button onClick={onClose} style={{ alignSelf: 'center' }}>Close</button>
    </div>
  </div>
);

const Spinner = () => (
  <div className="spinner-container">
    <div className="spinner"></div>
  </div>
);


const ProgressBar = ({ submittedCount, totalQuestions }) => (
  <div className="progress-bar">
    <div className="progress" style={{ width: `${(submittedCount / totalQuestions) * 100}%` }}></div>
  </div>
);

function QuizComponent({ quizType }) {
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [finished, setFinished] = useState(false);
  const [submittedCount, setSubmittedCount] = useState(0);
  const [Wrong, addWrong] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchQuestions();
  }, [quizType]);

  const fetchQuestions = async () => {
    setLoading(true);
    let url = `https://klingonapi-cafaedb94044.herokuapp.com/${quizType}-questions`;
    try {
      const response = await fetch(url);
      const data = await response.json();
      setQuestions(data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching questions:', error);
      setLoading(false);
    }
  };

  const handleAnswerSubmit = (isCorrect,remainingTime) => {
    const currentQuestion = questions[currentQuestionIndex];
    if (!isCorrect) {
      addWrong([...Wrong, {
        question: currentQuestion.question,
        correctOption: currentQuestion.options[currentQuestion.correct_index]
      }]);
    }
    if (isCorrect) {
      setScore(prevScore => prevScore + remainingTime); 
    }
    if (currentQuestionIndex === questions.length - 1) {
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
    setSubmittedCount(0);
    fetchQuestions();
    addWrong([]);
  };

  if (loading) {
    return <Spinner />;
  }

  return (
    <div>
      <button className="back-button" onClick={() => navigate('/quiz')}>
        <Icon icon={arrowBack} className="back-icon" />
      </button>
      <div>
        {!finished ? (
          <div className="quiz-container">
            <ProgressBar submittedCount={submittedCount} totalQuestions={questions.length} />
            <div className="quiz-info">
              <div className="score-tracker">Score: {score}</div>
            </div>
            {questions.length > 0 && (
              <QuizQuestion
                question={questions[currentQuestionIndex].question}
                options={questions[currentQuestionIndex].options}
                correctIndex={questions[currentQuestionIndex].correct_index}
                onAnswerSubmit={handleAnswerSubmit}
                onNextQuestion={handleNextQuestion}
                currentNumber={currentQuestionIndex}
                totalQuestions={questions.length}
                onTimeUp={() => setShowModal(true)}
              />
            )}
            {showModal && <TimeUpModal onClose={() => setShowModal(false)} />}
          </div>
        ) : (
          <QuizSummary score={score} onRestartQuiz={handleRestartQuiz} Wrong={Wrong} quizType={quizType} />
        )}
      </div>
    </div>
  );
}

export default QuizComponent;
