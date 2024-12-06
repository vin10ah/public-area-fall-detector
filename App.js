import React, { useState } from 'react';
import './App.css';

function App() {
    const [videoSrc, setVideoSrc] = useState(null);

    const handleUpload = async (event) => {
        const file = event.target.files[0];
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("http://localhost:5000/video_feed", {
            method: "POST",
            body: formData
        });

        setVideoSrc("http://localhost:5000/video_feed");
    };

    return (
        <div className="App">
            <h1>YOLO Object Detection</h1>
            <input type="file" onChange={handleUpload} />
            {videoSrc && <img src={videoSrc} alt="Processed Video" />}
        </div>
    );
}

export default App;
