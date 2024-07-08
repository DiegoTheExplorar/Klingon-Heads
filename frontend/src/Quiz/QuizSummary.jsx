import React from 'react';
import './QuizSummary.css';

function QuizSummary({ score,onRestartQuiz, Wrong }) {
  const performanceAnalysis = score / 120 >= 0.8 ? 'Great job!' : 'Keep practicing!';

  const uniqueWrongAnswers = Wrong.reduce((unique, item) => {
    if (!unique.some(obj => obj.question === item.question)) {
      unique.push(item);
    }
    return unique;
  }, []);

  return (
    <div className="quiz-summary">
      <h2>Quiz Completed!</h2>
      <p>Your final score is {score} out of {120}.</p>
      <p>{performanceAnalysis}</p>
      {uniqueWrongAnswers.length > 0 && (
        <div>
          <h3>Incorrectly Answered Questions:</h3>
          <ul>
            {uniqueWrongAnswers.map((item, index) => (
              <li key={index}>
                <p>Question: {item.question}</p>
                <p>Correct Answer: {item.correctOption}</p>
              </li>
            ))}
          </ul>
        </div>
      )}
      <button onClick={onRestartQuiz} className="restart-quiz-button">
        Restart Quiz
      </button>
    </div>
  );
}

export default QuizSummary;
