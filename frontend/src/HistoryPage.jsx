import React from 'react';

function HistoryPage({ history }) {
  return (
    <div className="container">
      <h1>Translation History</h1>
      <table>
        <thead>
          <tr>
            <th>Input</th>
            <th>Translation</th>
          </tr>
        </thead>
        <tbody>
          {history.map((item, index) => (
            <tr key={index}>
              <td>{item.input}</td>
              <td>{item.translation}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default HistoryPage;
