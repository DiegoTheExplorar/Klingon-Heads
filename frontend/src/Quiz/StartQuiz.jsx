import React from 'react';
import './StartQuiz.css';

function StartQuiz({ onStartQuiz }) {
  return (
    <div className="start-quiz-container">
      <h1>Welcome to the Quiz!</h1>
      <p>Test your knowledge and see how much you can score.</p>
      <button onClick={onStartQuiz} className="start-quiz-button">Start Quiz</button>
    </div>
  );
}

export default StartQuiz;
