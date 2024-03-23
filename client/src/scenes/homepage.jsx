import Navbar from "./navbar";
// import HeroSectionImage from "./../images/heroSection.jpeg";
import { useNavigate } from "react-router-dom";
import { useSelector } from 'react-redux';
// import Footer from "./footer";
// import Phone from "./phone";
import { toast } from "react-toastify";
import axios from "axios";
import { useState } from "react";


const Homepage = ()=>{
    const user = useSelector((state) => state.doc.user);
    const navigate = useNavigate();
    const [file, setFile] = useState();
    const [result, setResult] = useState();
    const [accuracy,setAccuracy] = useState();
    const [no_of_frames,setFrames] = useState();
    if(!user){
        setTimeout(() => {
            toast.error("Sign in again...")
            navigate("/");
        }, 2000);
    }

    const upload = ()=>{
        const formData = new FormData();
        formData.append('file', file);
        axios.post("https://28fe-103-124-122-210.ngrok-free.app/predict", formData)
        .then(res => {
            console.log(res);
            setResult(res.data.prediction);
            setAccuracy(res.data.accuracy);
            setFrames(res.data.no_of_frames);
        })
        .catch(er => console.log(er))
    }
   
    if( user ){
        return(
            <>
            <Navbar />
            <div className="">
                <div className="hero min-h-screen">
                    <div className="hero-overlay bg-opacity-60"></div>
                    <div className="hero-content text-center text-neutral-content">
                        <div className="max-w-md">
                        <h1 className="mb-20 text-4xl">Upload video here to check for deepfake</h1>
                        <input type="file" onChange={(e)=> {
                            setFile(e.target.files[0]);
                        }
                        }/>
                        <button type="button" className="border mt-20" onClick={upload}>Upload</button>
                        </div>
                    </div>
                    <div className="text-2xl">Result : {result}<br/>Accuracy of the Result : {accuracy}%<br/>Number of Frames  : {no_of_frames}</div>
                    
                    
                </div>
            </div>
            </>
        )   
    }
}

export default Homepage;