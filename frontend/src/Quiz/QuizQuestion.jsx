import React, { useEffect, useState } from 'react';
import './QuizQuestion.css';

function QuizQuestion({ question, options, onAnswerSubmit, currentNumber, totalQuestions, onNextQuestion }) {
  const [selectedOption, setSelectedOption] = useState('');
  const [submitted, setSubmitted] = useState(false);

  
  const handleSubmit = () => {
    if (!submitted) {
      setSubmitted(true); 
      onAnswerSubmit(selectedOption === options[0]);
    } else {
      onNextQuestion(); 
    }
  };

  useEffect(() => {
    setSelectedOption('');
    setSubmitted(false);
  }, [question]);

  return (
    <div className="question-container">
      <div className="question-box">
        <h3>{question}</h3>
        {options.map((option, index) => (
          <div key={index} className={`option ${selectedOption === option ? 'selected' : ''} ${submitted && (option === options[0] ? 'correct' : (option === selectedOption ? 'incorrect' : ''))}`}>
            <button onClick={() => !submitted && setSelectedOption(option)} disabled={submitted} className={selectedOption === option ? 'selected' : ''}>
              <span className="option-label">{String.fromCharCode(65 + index)}</span> {option}
            </button>
          </div>
        ))}
        <button onClick={handleSubmit} className="submit-button">
          {submitted ? (currentNumber + 1 === totalQuestions ? 'See Summary' : 'Next Question') : 'Submit'}
        </button>
      </div>
    </div>
  );
}

export default QuizQuestion;


