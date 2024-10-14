const chatContainer = document.getElementById("chat-container");
const chatBox = document.getElementById("chat-box");
const chatOptions = document.getElementById("chat-options");
const userInput = document.getElementById("user-input");

// Initialize a variable to store the section ID
// Initialize a variable to store the section ID
let sectionId = "";

// Function to generate a random section ID
function generateSectionId() {
  return 'section-' + Math.random().toString(36).substr(2, 9); // Generates a random alphanumeric ID
}

// Ensure section ID is generated when the chatbot opens
function initializeChatbot() {
  if (!sectionId) {
    sectionId = generateSectionId(); // Generate and assign section ID
    console.log("Section ID generated:", sectionId); // Debugging log
  }
}
// Predefined bot responses with keywords
const botResponses = {
  "hi": {
    "response": "Hi! Welcome to Terralogic, a global leader in IT solutions and digital transformation. How can I assist you today?",
    "options": ["tell me about your services", "who are your clients?", "what are your achievements?"]
  },
  "hello": {
    "response": "Hello! Welcome to Terralogic. How can I help you today?",
    "options": ["tell me about your services", "who are your clients?", "what are your achievements?"]
  },
  "tell me about your services": {
    "response": "We offer a wide range of services including Design, Application Development, IT Services, and Cybersecurity. Which one would you like to explore?",
    "options": ["explore design services", "explore application development", "explore it services", "explore cybersecurity"]
  },
  "explore design services": {
    "response": "Our design services cover UX Research, UI/UX Design, and Usability Testing. We’ve helped companies like Paytm and IMS with their digital transformation. What would you like to know more about?",
    "options": ["learn about ux research", "learn about ui/ux design", "learn about usability testing"]
  },
  "learn about ux research": {
    "response": "Our UX Research provides deep insights to optimize user experiences. We conduct Heuristic Analysis, Design Audits, and Usability Tests.",
    "options": ["explore more services", "explore case studies"]
  },
  "learn about ui/ux design": {
    "response": "We craft intuitive, user-centered designs that are scalable. From mobile apps to enterprise solutions, we design experiences users love.",
    "options": ["explore more design services", "tell me about your clients"]
  },
  "learn about usability testing": {
    "response": "We help improve product usability through rigorous testing to ensure a seamless user experience.",
    "options": ["explore application development", "tell me about your achievements"]
  },
  "explore application development": {
    "response": "Our application development covers mobile, web, and enterprise-level apps, utilizing cutting-edge technologies. What specific area interests you?",
    "options": ["mobile app development", "web app development", "enterprise app development"]
  },
  "mobile app development": {
    "response": "We build feature-rich, high-performance mobile apps tailored to your needs, for both Android and iOS platforms.",
    "options": ["explore more about app development", "who are your mobile clients?"]
  },
  "web app development": {
    "response": "Our web app solutions provide robust and scalable platforms for businesses, focusing on user experience and performance.",
    "options": ["learn about custom web apps", "who are your clients?"]
  },
  "enterprise app development": {
    "response": "We develop enterprise applications that streamline processes and increase efficiency. Our clients include Fortune 500 companies.",
    "options": ["learn about saas solutions", "explore more services"]
  },
  "explore it services": {
    "response": "We offer 24/7 IT support, managed services, and infrastructure outsourcing to help businesses stay agile and secure.",
    "options": ["learn about managed services", "learn about cloud services", "learn about infrastructure outsourcing"]
  },
  "learn about managed services": {
    "response": "Our managed services provide proactive support and monitoring, ensuring that your IT infrastructure is secure and efficient.",
    "options": ["learn about cloud security", "explore more services"]
  },
  "learn about cloud services": {
    "response": "We provide cloud hosting, DaaS, and cloud management services for businesses looking to modernize and scale their operations.",
    "options": ["explore application hosting", "learn about cloud security"]
  },
  "learn about cloud security": {
    "response": "We offer robust cloud security services that ensure your cloud infrastructure is protected from threats. This includes encryption, identity management, and real-time monitoring.",
    "options": ["learn about data protection", "explore compliance services"]
  },
  "learn about infrastructure outsourcing": {
    "response": "We offer infrastructure outsourcing, including network management and IT support, enabling businesses to focus on growth.",
    "options": ["explore more it services", "learn about system integration"]
  },
  "explore cybersecurity": {
    "response": "We provide cutting-edge cybersecurity solutions including Managed Detection, Threat Management, and Governance, Risk & Compliance.",
    "options": ["learn about threat management", "learn about data protection", "learn about iot/ot security"]
  },
  "learn about threat management": {
    "response": "Our threat management services provide proactive defense against cyber threats, using AI-driven tools and human expertise.",
    "options": ["learn about vulnerability assessments", "explore more cybersecurity services"]
  },
  "learn about data protection": {
    "response": "We ensure data privacy and protection with secure solutions for data at rest and in motion, ensuring compliance with global standards.",
    "options": ["explore compliance services", "learn about iot/ot security"]
  },
  "learn about iot/ot security": {
    "response": "We secure IoT and OT devices with real-time monitoring and threat mitigation to ensure the safety of interconnected systems.",
    "options": ["explore cybersecurity solutions", "learn about managed services"]
  },
  "who are your clients?": {
    "response": "We are proud to have worked with companies like Upstox, Paytm, IMS Mobile, and Trukkin, across industries like Fintech, Healthcare, and Agritech.",
    "options": ["learn more about fintech clients", "learn more about healthcare clients", "tell me about your achievements"]
  },
  "learn more about fintech clients": {
    "response": "In the Fintech space, we've partnered with companies like Paytm and Upstox to revolutionize digital finance platforms.",
    "options": ["learn about fintech solutions", "explore more clients"]
  },
  "learn more about healthcare clients": {
    "response": "We've developed solutions for the Healthcare industry, including telemedicine platforms and wellness apps.",
    "options": ["learn about healthcare solutions", "explore more industries"]
  },
  "what are your achievements?": {
    "response": "We have received several accolades, including the DNA Paris Design Award and A'Design Award. We’ve completed over 2400 projects across 16+ countries.",
    "options": ["explore more achievements", "learn more about your services"]
  },
  "bye": {
    "response": "Goodbye! We hope to assist you again soon. Have a great day!",
    "options": []
  }
};


// Function to add a message to the chat
function addMessage(sender, text) {
  const messageElement = document.createElement("div");
  messageElement.classList.add("chat-message", sender);
  messageElement.textContent = text;
  chatBox.appendChild(messageElement);
  scrollToBottom();
}

// Function to show button options for user to select
function showOptions(options) {
  chatOptions.innerHTML = ""; // Clear previous options
  options.forEach(option => {
    const button = document.createElement("button");
    button.classList.add("option-button");
    button.textContent = option;
    button.onclick = () => handleOptionClick(option);
    chatOptions.appendChild(button);
  });
}

// Handle button click and bot reply
function handleOptionClick(option) {
  addMessage("user", option);
  chatOptions.innerHTML = ""; // Hide options after user clicks one
  botReply(option);
}

// Function to handle user-typed message
function sendMessage() {
  const userText = userInput.value.trim().toLowerCase();
  if (userText === "") return;

  addMessage("user", userText);
  userInput.value = ""; // Clear input field

  // Check for matches with keywords in botResponses
  const matchedOptions = Object.keys(botResponses).filter(key => userText.includes(key));
  if (matchedOptions.length > 0 && botResponses[matchedOptions[0]].response === ""){
    makeHttpRequest(userText);
    botReply(matchedOptions[0]);
  }
  else if (matchedOptions.length > 0 && botResponses[matchedOptions[0]].response !== "") {
    // If there's a match, respond with the first match found
    
    botReply(matchedOptions[0]);
  } else {
    // If no matches, make an HTTP request
    makeHttpRequest(userText);
  }
}

// Handle pressing Enter key
function handleKeyPress(event) {
  if (event.key === "Enter") {
    sendMessage();
  }
}

// Function to generate bot reply based on user input
function botReply(userText) {
  setTimeout(() => {
    // Check for predefined responses
    const botResponse = botResponses[userText];
    

    if (botResponse) {
      if(botResponse.response !== "") addMessage("bot", botResponse.response);

      // If there are options, show them
      if (botResponse.options.length > 0) {
        showOptions(botResponse.options);
      }
    } else {
      addMessage("bot", "I didn't understand that. Could you please rephrase?");
    }
  }, 1000); // Simulate bot thinking time
}

// Function to make HTTP request for user input
async function makeHttpRequest(userText) {
  // Ensure section ID is generated
  if (!sectionId) {
    sectionId = generateSectionId(); // Generate and assign section ID if not already generated
  }

  try {
    const response = await fetch('http://192.168.55.89:5000/ask', { // Change to your API endpoint
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        user_input: userText,
        section_id: sectionId, // Pass the section ID with the request
        model_choice:"google"
      })
    });

    // Check if the response is OK
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    // Convert the response to JSON
    const data = await response.json();

    // Access the 'response' property correctly
    if (data.response) {
      addMessage("bot", data.response); // Display the bot's response
    } else {
      addMessage("bot", "I did not get a valid response from the server.");
    }
  } catch (error) {
    console.error('Error:', error); // Log the error for debugging
    addMessage("bot", "Sorry, I could not reach the server. Please try again later.");
  }
}

// Scroll to the bottom of the chat
function scrollToBottom() {
  chatBox.scrollTop = chatBox.scrollHeight;
}

function toggleChatbot() {
  const chatContainer = document.getElementById("chat-container");

  if (chatContainer.style.display === "none") {
    chatContainer.style.display = "flex";
    parent.postMessage({ action: "expand" }, "*"); // Notify parent to expand iframe
  } else {
    chatContainer.style.display = "none";
    parent.postMessage({ action: "collapse" }, "*"); // Notify parent to collapse iframe
  }
}

// Event listener for Enter key
userInput.addEventListener("keypress", handleKeyPress);