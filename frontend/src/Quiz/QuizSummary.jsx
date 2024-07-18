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

  const performanceAnalysis = score / 60 >= 0.8 ? 'Great job!' : 'Keep practicing!';

  const uniqueWrongAnswers = Wrong.reduce((unique, item) => {
    if (!unique.some(obj => obj.question === item.question)) {
      unique.push(item);
    }
    return unique;
  }, []);

  return (
    <div className="quiz-summary">
      <h2 style={{fontSize: '64px', fontWeight: 'bold', textAlign: 'center'}}>Completed!</h2>
      <h3 style={{fontSize: '24px'}}>Your final score is {score} out of 60.</h3>
      <h3 style={{fontSize: '24px'}}>{performanceAnalysis}</h3>
      {uniqueWrongAnswers.length > 0 && (
        <div>
          <ul>
            {uniqueWrongAnswers.map((item, index) => (
              <li key={index}>
                <p style={{color: 'red'}}>Question: {item.question}</p>
                <p style={{color: 'green'}}>Correct Answer: {item.correctOption}</p>
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
