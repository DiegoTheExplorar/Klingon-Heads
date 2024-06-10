// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDRYLgutyR26G-i48opAnjxik0jEuP6FxI",
  authDomain: "klingonheads.firebaseapp.com",
  projectId: "klingonheads",
  storageBucket: "klingonheads.appspot.com",
  messagingSenderId: "315030554218",
  appId: "1:315030554218:web:8ef748fc86fb7a37d76c13",
  measurementId: "G-BRF0VBH0KB"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = firebase.auth();
export default firebase;