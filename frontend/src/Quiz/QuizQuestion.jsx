import React, { useEffect, useState } from 'react';
import './QuizQuestion.css';
import { Icon } from '@iconify/react';
import timerIcon from '@iconify-icons/mdi/timer';

const time = 15;

function QuizQuestion({ question, options, correctIndex, onAnswerSubmit, onNextQuestion, currentNumber, totalQuestions, onTimeUp }) {
  const [selectedOption, setSelectedOption] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [timer, setTimer] = useState(time);
  useEffect(() => {
    const timerId = setInterval(() => {
      setTimer(prevTimer => {
        if (prevTimer > 1) return prevTimer - 1;
        clearInterval(timerId);
        if (!isSubmitted) {
          onTimeUp();
          onAnswerSubmit(false,0);
          setIsSubmitted(true);
        }
        return 0;
      });
    }, 1000);

    return () => clearInterval(timerId);
  }, [isSubmitted]);

  useEffect(() => {
    setSelectedOption('');
    setIsSubmitted(false);
    setTimer(time); 
  }, [question]);

  const handleOptionSelect = option => {
    if (!isSubmitted) {
      setSelectedOption(option);
    }
  };

  const handleSubmit = () => {
    if (!isSubmitted) {
      setIsSubmitted(true);
      onAnswerSubmit(selectedOption === options[correctIndex],timer);
      setTimer(0);
    } else {
      onNextQuestion();
    }
  };

  return (
    <div>
    <div className="timer">
      <Icon icon={timerIcon} className="timer-icon" />
      <h3>{timer}s</h3> 
    </div>
    <div className="question-container">
      <div className="question-box">
      <h3 className="question">{question}</h3>
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
    </div>
  );
}

export default QuizQuestion;

