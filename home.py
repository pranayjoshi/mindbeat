import streamlit as st

def navigate_to_dashboard():
    st.session_state.page = 'dashboard'

def Home():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

            .main {
                background-color: #0A0A0A;
                color: #FFFFFF;
                font-family: 'Inter', sans-serif;
            }

            .top-left {
                position: absolute;
                top: 20px;
                left: 20px;
                display: flex;
                align-items: center;
            }
            .top-left img {
                height: 40px;
                margin-right: 10px;
            }
            .top-left .name {
                font-size: 1.5rem;
                font-weight: 700;
                color: #FFFFFF;
            }

            .center-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: calc(100vh - 76px);
                padding: 2rem;
                text-align: center;
                position: relative;
                overflow: hidden;
            }

            .header2 {
                font-size: 3.8rem;
                font-weight: 800;
                background: linear-gradient(90deg, #9333E0, #C026D3);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1.5rem;
            }

            .subheader {
                font-size: 1.2rem;
                color: #9CA3AF;
                margin-bottom: 2rem;
                max-width: 48rem;
                line-height: 1.5;
            }

            .floating-element {
                position: absolute;
                background-color: rgba(147, 51, 234, 0.1);
                border-radius: 50%;
                animation: float 15s infinite;
            }

            @keyframes float {
                0% { transform: translate(0, 0) rotate(0deg); }
                33% { transform: translate(30px, -50px) rotate(120deg); }
                66% { transform: translate(-20px, 20px) rotate(240deg); }
                100% { transform: translate(0, 0) rotate(360deg); }
            }

            /* Align Streamlit button */
            div.stButton > button {
                background: linear-gradient(90deg, #9333EA 0%, #C026D3 100%);
                color: white;
                border-radius: 8px;
                padding: 0.75rem 2rem;
                border: none;
                font-size: 1.125rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
                margin-top: 10px;
                box-shadow: 0 4px 14px rgba(147, 51, 234, 0.3);
            }
            div.stButton > button:hover {
                opacity: 0.9;
                transform: translateY(-2px);
            }
        </style>
        
        <div class="top-left">
            <img src="https://github.com/pranayjoshi/mind_music/blob/main/log.png?raw=true">
            <div class="header2">MindBeats</div>
        </div>
        
        <div class="center-container">
            <div class="floating-element" style="width: 100px; height: 100px; top: 10%; left: 10%;"></div>
            <div class="floating-element" style="width: 50px; height: 50px; top: 20%; right: 20%;"></div>
            <div class="floating-element" style="width: 75px; height: 75px; bottom: 15%; left: 15%;"></div>
            <div class="header2">Neuroadaptive Music Recommendation</div>
            <div class="subheader">Experience the future of music with MindBeats.
            Our BCI headset reads your emotions in real-time, creating perfectly curated playlists that match your mood.
            As your emotions evolve, our AI adapts your music dynamically, learning your preferences to deliver an ever-improving, personalized listening experience.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Place the button right below the content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Started ⚡"):
            navigate_to_dashboard()
