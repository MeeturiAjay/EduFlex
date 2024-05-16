import json
import urllib.parse
import requests
import streamlit as st
from pytube import YouTube
from streamlit_clickable_images import clickable_images
from streamlit_option_menu import option_menu
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import YoutubeLoader
from apikey import geminiai_api_key
apikey = geminiai_api_key
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import YoutubeLoader
from apikey import geminiai_api_key

os.environ["GOOGLE_API_KEY"] = geminiai_api_key

genai.configure(api_key=apikey)

chosen = option_menu(
    menu_title=None,
    options = ["Welcome","Learn","Skill Test","Road Map"],
    icons = ["snow3","bi-book-half","clipboard2-data","clipboard2-check"],
    default_index=0,
    orientation="horizontal")

if chosen == "Welcome":
    st.title("Welcome to :blue[EduFlex]: Your Personalized Learning Companion 📚 ")
    from streamlit_lottie import st_lottie
    import requests
    import json

    def load_lottie(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    def load_local_lottie(filepath):
        with open(filepath,"r") as f:
            return json.load(f)

    lottie_animation1 = "https://lottie.host/cca84fef-b871-48c8-a756-ab0f3f17fcd5/hjJGMFluMT.json"
    lottie_anime_json = load_lottie(lottie_animation1)

    col1,col2 = st.columns(2)
    with col1:
        st_lottie(lottie_anime_json,key="student",height=400,width=400)
    with col2:

        # Display bullet points using HTML syntax
        st.markdown("""
        <ul>
          <li style="font-size:25px ; margin-top:35px">At EduFlex, we revolutionize the way you learn by tailoring every aspect of your educational journey to your unique needs and preferences. Our comprehensive platform offers a seamless blend of innovative features designed to maximize your learning potential and achieve your academic goals effortlessly.</li>
        </ul>
        """, unsafe_allow_html=True)
    col3,col4 = st.columns(2)
    with col4:
        lottie_animation2 = load_local_lottie("animation.json")
        st_lottie(lottie_animation2,key="study",height=400,width=400)
    with col3:
        import streamlit as st

        # Display bullet points using HTML syntax
        st.markdown("""
        <ul>
          <li style="font-size:23px ; margin-top:80px">Personalized Study Plans</li>
          <li style="font-size:23px">Curated Video Tutorials</li>
          <li style="font-size:23px">Focused Learning Environment</li>
          <li style="font-size:23px">Pomodoro Technique Integration</li>
          <li style="font-size:23px">Interactive Chat Support</li>
          <li style="font-size:23px">Skill Testing and Assessment</li>
        </ul>
        """, unsafe_allow_html=True)

    col5 , col6 = st.columns(2)

    with col6:
        st.markdown("""
                <ul>
                  <li style="font-size:25px ; margin-top:35px"><h3>Personalized Study Plans</h3> Say goodbye to one-size-fits-all approaches! With EduFlex, we understand that every learner is different. That's why we gather essential information from you and craft a customized study plan meticulously tailored to your schedule and learning objectives. Whether you're a beginner, intermediate, or advanced learner, we've got you covered.</li>
                </ul>
                """, unsafe_allow_html=True)
    with col5:
        lottie_animation2 = load_local_lottie("animation1.json")
        st_lottie(lottie_animation2, key="courseplan", height=450, width=450)

    col7,col8 = st.columns(2)
    with col7:
        st.markdown("""
                <ul>
                    <li style="font-size:25px ; margin-top:35px"><h3>Curated Video Tutorials :</h3> Enhance your learning experience with our curated selection of video tutorials. Based on your study plan and skill level, we recommend relevant video resources to supplement your learning journey. With EduFlex, you'll have access to high-quality educational content that's both engaging and informative.</li>
                </ul>
                """, unsafe_allow_html=True)
    with col8:
        lottie_animation2 = load_local_lottie("animation2.json")
        st_lottie(lottie_animation2, key="video", height=400, width=400)

    col9,col10 = st.columns(2)
    with col10:
       st.markdown("""
                <ul>
                  <li style="font-size:25px ; margin-top:35px"><h3>Focused Learning Environment: </h3> We believe in the power of focus. That's why EduFlex integrates a unique feature that blocks irrelevant tabs and distractions while you're engaged in your learning sessions. Say hello to a distraction-free environment where you can concentrate fully on mastering new concepts.</li>
               </ul>
               """, unsafe_allow_html=True)
    with col9:
        lottie_animation2 = load_local_lottie("animation3.json")
        st_lottie(lottie_animation2, key="focus", height=400, width=400)
    col11, col12 = st.columns(2)
    with col11:
       st.markdown("""
                <ul>
                  <li style="font-size:25px ; margin-top:35px"><h3>Pomodoro Technique Integration:</h3> Boost your productivity and maintain peak performance with the Pomodoro Technique. Our extension includes built-in Pomodoro timers, allowing you to structure your study sessions effectively and optimize learning intervals with short breaks for improved retention.</li>
               </ul>
               """, unsafe_allow_html=True)
    with col12:
        lottie_animation2 = load_local_lottie("animation4.json")
        st_lottie(lottie_animation2, key="pomodoro", height=400, width=400)

    col13,col14 = st.columns(2)
    with col14:
       st.markdown("""
                <ul>
                  <li style="font-size:25px ; margin-top:35px"><h3> Interactive Chat Support:</h3> Need clarification on a concept? No problem! With our interactive chat support feature, you can engage in real-time conversations with our intelligent chatbot while watching video tutorials. Ask questions, seek guidance, and deepen your understanding—all within the EduFlex platform.</li>
               </ul>
               """, unsafe_allow_html=True)
    with col13:
        lottie_animation2 = load_local_lottie("animation5.json")
        st_lottie(lottie_animation2, key="chatbot", height=400, width=400)

    col15,col16 = st.columns(2)
    with col15:
       st.markdown("""
                <ul>
                  <li style="font-size:25px ; margin-top:35px"><h3>Skill Testing and Assessment:</h3> Measure your progress and reinforce your learning with our skill testing feature. After completing a video tutorial, take a skill test to evaluate your comprehension and identify areas for improvement. At EduFlex, we're committed to helping you enhance your understanding and achieve academic success.</li>
               </ul>
               """, unsafe_allow_html=True)
    with col16:
        lottie_animation2 = load_local_lottie("animation6.json")
        st_lottie(lottie_animation2, key="skill-test", height=400, width=400)



    st.markdown("<h1 style='color: #6236ea; text-align:center'>Join EduFlex Today!</h1>", unsafe_allow_html=True)

if chosen == "Learn":

    st.title("Course recommender")
    # Replace 'YOUR_API_KEY' with your actual API key
    API_KEY = "AIzaSyAMcfmKXIcfVNjRFMhAEGANXxNQ6r6wJzc"


    @st.cache_data
    # Function to search for videos based on a query
    def search_videos(query, max_results=6):
        url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults={max_results}&q={urllib.parse.quote(query)}&type=video&key={API_KEY}"
        response = requests.get(url)
        data = response.json()
        video_links = [f"https://www.youtube.com/watch?v={item['id']['videoId']}" for item in data['items']]
        return video_links

    # Ask user for search query
    query = st.text_input("What do you want to learn")
    level = st.selectbox("Choose your learning level", ["Beginner", "Intermediate", "Advanced"])

    full_query = f"{query} {level} full course"
    videos = search_videos(full_query)
    links = []
    #print("YouTube Links:")
    for video in videos:
        links.append(video)

    @st.cache_data
    def youtube(url):
        yt = YouTube(url)
        return yt.title,yt.thumbnail_url

    titles = []
    locations=[]
    thumbnails = []

    for video_url in links:
        video_title , video_thumbnail = youtube(video_url)
        titles.append(video_title)
        thumbnails.append(video_thumbnail)
    if query:
        selected_video = clickable_images(thumbnails,
                        titles=titles,div_style={"height": "500px","display": "grid", "grid-template-columns":"repeat(2,1fr)" ,"justify-content": "center","align-item":"center",
                        "flex-wrap": "nowrap", "overflow-y": "auto"},
                        img_style={"margin": "10px", "height": "150px"})
        st.markdown(f"Thumbnail #{selected_video} clicked" if selected_video >-1 else None)
        if selected_video > -1:
            video_url = links[selected_video]
            video_title = titles[selected_video]

            st.header(video_title)
            st.video(video_url)

            if "notes" not in st.session_state:
                st.session_state["notes"] = []


            def add_note(note_text):
                if note_text:
                    st.session_state["notes"].append(note_text)
                    st.success("Note added successfully!")
                else:
                    st.warning("Please enter a note before adding.")


            def display_notes():
                if st.session_state["notes"]:
                    st.header("Your Notes:")
                    for note in st.session_state["notes"]:
                        st.write(note)
                    st.button("Clear Notes", on_click=st.session_state["notes"].clear)  # Clear notes on click
                else:
                    st.info("No notes added yet.")


            st.header("Take Notes")
            note_text = st.text_area("Write a note:", height=100)
            add_note_button = st.button("Add Note")

            if add_note_button:
                add_note(note_text)
                st.empty()  # Clear note input for better UX

            display_notes_button = st.button("Display All Notes")
            if display_notes_button:
                display_notes()
            col1,col2=st.columns(2)
            with col2:
                chat_with_video = option_menu(
                                menu_title=None,
                                options = ["Chat with Video"],
                                icons = ['alexa'],
                                default_index=0,
                                )
            if chat_with_video:


                apikey = geminiai_api_key
                from youtube_transcript_api import YouTubeTranscriptApi

                os.environ["GOOGLE_API_KEY"] = geminiai_api_key

                genai.configure(api_key=apikey)


                @st.cache_data
                def get_video_text(video_url):
                    try:
                        video_id = video_url.split("=")[1]
                        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

                        transcript = ""
                        for i in transcript_text:
                            transcript += " " + i['text']

                        return transcript

                    except Exception as e:
                        raise e


                def get_pdf_text(pdf_docs):
                    text = ""
                    for pdf in pdf_docs:
                        pdf_reader = PdfReader(pdf)
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                    return text


                @st.cache_data
                def get_text_chunks(text):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                    chunks = text_splitter.split_text(text)
                    return chunks


                @st.cache_data

                def get_vector_store(text_chunks):
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                    vector_store.save_local("faiss_index")


                @st.cache_data
                def get_conversational_chain():
                    prompt_template = '''Answer the question as detailed as possible from the provided context, make sure to provide all the details.
                    Context:\n {context}?\n
                    Question: \n{question}\n
    
                    Answer:
                    '''
                    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

                    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'questions'])
                    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

                    return chain


                @st.cache_data
                def user_input(user_question):
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

                    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(user_question)

                    chain = get_conversational_chain()

                    response = chain({
                        "input_documents": docs, "question": user_question
                    },
                        return_only_outputs=True)

                    print(response)
                    st.write("Reply : ", response["output_text"])


                def main():
                    #st.set_page_config("Chat PDF")
                    st.header("Chat with Youtube Video💁")

                    user_question = st.text_input(f"Ask a Question from the {video_title} ")

                    if user_question:
                        user_input(user_question)



                    #video_url = st.text_input("Upload the youtube video link here")
                    #if st.button("Submit & Process"):
                    with st.spinner("Processing..."):
                        raw_text = get_video_text(video_url)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")


                if __name__ == "__main__":
                    main()


if chosen == "Skill Test":

    os.environ["GOOGLE_API_KEY"] = "AIzaSyCswPVrlnUZoWLig2VlF3Hj0p7pJBuOoTE"
    genai.configure(api_key="AIzaSyCswPVrlnUZoWLig2VlF3Hj0p7pJBuOoTE")
    st.title("Start taking Skill Test from the skills you have learnt")

    skill = st.text_input("Enter the skill that you want to take the test")
    skill_level = st.selectbox("At which level are you in the skill",["Beginner","Intermediate","Advanced"])
    skill_topic = st.text_input("On what topic in skill you want to take on?")
    number_of_questions = st.number_input("Choose number of questions",value=1)
    number_of_questions = str(number_of_questions)
    submit = st.button("submit")

    prompt = f'''You are a multiple choice questions generator. I want you to prepare a multiple choice
                quiz on the {skill_topic} topic of the chosen {skill} skill. The questions need to match the {skill_level} level.
                generate {number_of_questions} questions.
                The output should be in the format of :
                Which of these is correct declaration of a variable in python?
                a.vrae2Z
                b.2333yu
                c.var_name
                d.None. Strictly follow the same question pattern.
                After the end of the questions didplay the correct answers'''

    if submit:

        model = genai.GenerativeModel("gemini-pro")
        questions = model.generate_content(prompt+skill_topic+skill+skill_level+number_of_questions)

        st.markdown(questions.text)



if chosen == "Road Map":
    import streamlit as st
    import os
    import google.generativeai as genai
    from apikey import geminiai_api_key

    apikey = geminiai_api_key

    os.environ['GOOGLE_API_KEY'] = geminiai_api_key

    genai.configure(api_key=apikey)

    st.title("Daily Course Planner")

    st.text_input("Please tell us your name")

    col1, col2 = st.columns(2)

    with col1:
        course = st.text_input("Which course do you want to learn")

    with col2:
        learning_level = st.selectbox("Choose your level of learning", ["Beginner", "Intermediate", "Advanced"])

    col3, col4 = st.columns(2)

    with col3:
        no_of_hours = st.number_input("Choose number of learning hours per day", value=1, placeholder="Choose here")
    with col4:
        no_of_days = st.number_input("How many days you want to learn the skill", value=1, placeholder="Choose here")

    target_level = st.selectbox("To which level do you want to reach at the end of the preparation",
                                ["Intermediate", "Advanced"])

    no_of_days = str(no_of_days)
    no_of_hours = str(no_of_hours)

    submit = st.button("Submit")

    prompt_template = f'''You are course a lecture in the course.{course}I am you student who want to learn the course that you teach.
                            I am at this {learning_level}.I want to learn the course in {no_of_days} days by spending {no_of_hours} hours daily.
                            At the end of the preparation I need to reach {target_level}.As a lecturer in  the {course} I want a daily plan 
                            to follow to learn that course.Give me the plan according to the inputs that I have provided to reach the {target_level}.
                            Tell me where I have to start at the start of the day and the topics that I have to cover in the day and at what timings.
                            At the end of the day I want to revise the topics that  I have learnt at the end of the day so please add the revision time and the 
                            topics to recall. Start the plan with a motivational quote so that I can have a positive attitude while learning.Add the timings in which
                            to complete the topic.I cannot spend all {no_of_hours} hours at once so divide them in the day at each part of the day.

                            Output Format :
                            DAY_1 : 
                            9-10 : Topicsto Cover
                            10-11 : Topics to cover
                            ""
                            ""
                            '''
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(
        prompt_template + course + learning_level + no_of_hours + no_of_days + target_level)

    if submit:
        st.spinner(text="Processing...")
        st.markdown(response.text)




