import React, { useEffect, useState } from 'react';
import './QuizQuestion.css';

function QuizQuestion({ question, options, correctIndex, onAnswerSubmit, currentNumber, totalQuestions, onNextQuestion }) {
  const [selectedOption, setSelectedOption] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);

  useEffect(() => {
    setSelectedOption('');
    setIsSubmitted(false);
  }, [question]);

  const handleOptionSelect = option => {
    if (!isSubmitted) {
      setSelectedOption(option);
    }
  };

  const handleSubmit = () => {
    if (!isSubmitted) {
      setIsSubmitted(true);
      onAnswerSubmit(selectedOption === options[correctIndex]); 
    } else {
      onNextQuestion();
    }
  };

  return (
    <div className="question-container">
      <div className="question-box">
        <h3>{question}</h3>
        {options.map((option, index) => (
          <div key={index} className={`option ${selectedOption === option ? 'selected' : ''} ${isSubmitted && (index === correctIndex ? 'correct' : (selectedOption === option ? 'incorrect' : ''))}`}>
            <button onClick={() => handleOptionSelect(option)} disabled={isSubmitted} className={selectedOption === option ? 'selected' : ''}>
              <span className="option-label">{String.fromCharCode(65 + index)}</span> {option}
            </button>
          </div>
        ))}
        <button onClick={handleSubmit} className="submit-button">
          {isSubmitted ? (currentNumber + 1 === totalQuestions ? 'See Summary' : 'Next Question') : 'Submit'}
        </button>
      </div>
    </div>
  );
}


export default QuizQuestion;


