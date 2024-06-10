import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";

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
const auth = getAuth(app);
const provider = new GoogleAuthProvider();
export { auth, provider };
