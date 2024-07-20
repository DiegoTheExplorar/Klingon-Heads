import React, { useEffect } from 'react';
import './Modal.css';

const Modal = ({ children, showModal, onClose }) => {
  useEffect(() => {
    if (showModal) {
      const timer = setTimeout(() => {
        onClose();
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [showModal, onClose]);

  if (!showModal) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        {children}
      </div>
    </div>
  );
};

export default Modal;
