import React, { useEffect, useState } from 'react';
import { addHighScoreToFirestore, getHighScoreFromFirestore } from '../FireBase/firebasehelper';
import './QuizSummary.css';

function QuizSummary({ score, onRestartQuiz, Wrong, quizType }) {
  const [highScore, setHighScore] = useState(null);
  const [message, setMessage] = useState('');

  useEffect(() => {
    async function fetchHighScore() {
      try {
        const fetchedHighScore = await getHighScoreFromFirestore(quizType);
        if (fetchedHighScore) {
          setHighScore(fetchedHighScore.score);
          if (score > fetchedHighScore.score) {
            setMessage('Congratulations! You achieved a new high score!');
            await addHighScoreToFirestore(score, quizType);
          } else {
            setMessage('Good job! Try again to beat your high score.');
          }
        } else {
          setMessage('Congratulations! You set the first high score!');
          await addHighScoreToFirestore(score, quizType);
        }
      } catch (error) {
        console.error('Error fetching or updating high score:', error);
      }
    }
    fetchHighScore();
  }, [score, quizType]);

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
      <p>Your final score is {score} out of 120.</p>
      <p>{performanceAnalysis}</p>
      <p>{message}</p>
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
