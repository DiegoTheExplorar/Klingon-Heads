import React from 'react';
import './QuizSummary.css';

function QuizSummary({ score, totalQuestions, onRestartQuiz }) {
  const performanceAnalysis = score / totalQuestions >= 0.8 ? 'Great job!' : 'Keep practicing!';

  return (
    <div className="quiz-summary">
      <h2>Quiz Completed!</h2>
      <p>Your final score is {score} out of {totalQuestions}.</p>
      <p>{performanceAnalysis}</p>
      <button onClick={onRestartQuiz} className="restart-quiz-button">
        Restart Quiz
      </button>
    </div>
  );
}

export default QuizSummary;
