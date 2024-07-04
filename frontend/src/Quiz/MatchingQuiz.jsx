import React, { useState } from 'react';
import { DragDropContext, Draggable, Droppable } from 'react-beautiful-dnd';

function MatchingQuiz({ quizData }) {
    const [userAnswers, setUserAnswers] = useState({}); // To store user matches

    const onDragEnd = (result) => {
        const { source, destination } = result;
        if (!destination) {
            return;
        }
        if (destination.droppableId === source.droppableId && destination.index === source.index) {
            return;
        }
        setUserAnswers({
            ...userAnswers,
            [source.droppableId]: destination.droppableId
        });
    };

    const handleSubmit = () => {
        let score = 0;
        Object.keys(userAnswers).forEach(questionId => {
            const question = quizData.find(item => item.id.toString() === questionId);
            if (question && question.klingon === userAnswers[questionId]) {
                score += 1; // Increment score if matched correctly
            }
        });
        alert(`You got ${score} out of ${quizData.length} correct!`);
    };

    return (
        <DragDropContext onDragEnd={onDragEnd}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '20px' }}>
                <Droppable droppableId="english">
                    {(provided) => (
                        <div ref={provided.innerRef} {...provided.droppableProps} style={{ width: '40%' }}>
                            <h3>English</h3>
                            {quizData.map((item, index) => (
                                <Draggable key={item.id} draggableId={item.english} index={index}>
                                    {(provided) => (
                                        <p ref={provided.innerRef} {...provided.draggableProps} {...provided.dragHandleProps}>
                                            {item.english}
                                        </p>
                                    )}
                                </Draggable>
                            ))}
                            {provided.placeholder}
                        </div>
                    )}
                </Droppable>
                <Droppable droppableId="klingon">
                    {(provided) => (
                        <div ref={provided.innerRef} {...provided.droppableProps} style={{ width: '40%' }}>
                            <h3>Klingon</h3>
                            {quizData.map((item, index) => (
                                <div key={item.id} style={{ margin: '10px 0', padding: '5px', border: '1px solid gray' }}>
                                    {userAnswers[item.english] === item.klingon && <span>Matched!</span>}
                                    {item.klingon}
                                </div>
                            ))}
                            {provided.placeholder}
                        </div>
                    )}
                </Droppable>
            </div>
            <button onClick={handleSubmit}>Submit Answers</button>
        </DragDropContext>
    );
}

export default MatchingQuiz;
