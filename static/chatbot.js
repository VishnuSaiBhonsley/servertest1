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
    "response": "Hi there! Welcome to Lollypop - A Terralogic Company! How can I assist you today?",
    "options": ["tell me about your services", "who are your clients?", "what are your achievements?"]
  },
  "hello": {
    "response": "Hi there! Welcome to Lollypop - A Terralogic Company! How can I assist you today?",
    "options": ["tell me about your services", "who are your clients?", "what are your achievements?"]
  },
  "services": {
    "response": "We offer a range of services including Research, Design, and Build to create intuitive and scalable digital products. Our services include User Experience Design, Digital Branding, Front-End Development, and more.",
    "options": ["tell me more about research", "tell me more about design", "tell me more about build"]
  },
  "tell me about your services": {
    "response": "We offer a range of services including Research, Design, and Build to create intuitive and scalable digital products. Our services include User Experience Design, Digital Branding, Front-End Development, and more.",
    "options": ["tell me more about research", "tell me more about design", "tell me more about build"]
  },
  "research": {
    "response": "Our research services provide data-driven insights that represent the voice of the user and resonate with business objectives. We offer Heuristic Analysis, Design Audit, Usability Testing, and more.",
    "options": ["tell me about your design process"]
  },
  "tell me more about research": {
    "response": "Our research services provide data-driven insights that represent the voice of the user and resonate with business objectives. We offer Heuristic Analysis, Design Audit, Usability Testing, and more.",
    "options": ["tell me about your design process", "tell me about your build process"]
  },
  "tell me more about design": {
    "response": "Our design team creates user-centric designs that are intuitive and scalable. We focus on Digital Branding, User Experience Design, User Interface Design, and more.",
    "options": ["tell me about your build process"]
  },
  "tell me more about build": {
    "response": "We help translate designs into pixel-perfect, adaptable digital products. Our build services include Front-End Development, Web Applications, Mobile Applications, and Custom Applications.",
    "options": ["who are your clients?", "what are your achievements?"]
  },
  "about": {
    "response": "We help translate designs into pixel-perfect, adaptable digital products. Our build services include Front-End Development, Web Applications, Mobile Applications, and Custom Applications.",
    "options": ["who are your clients?", "what are your achievements?", "tell me more about your work","what industries do you work in?"]
  },
  "who are your clients?": {
    "response": "We are proud to have worked with clients like SBI Caps, Zudio, Paytm Money, and IMS Mobile. Our work spans across various industries including Healthcare, Fintech, and Agritech.",
    "options": ["what are your achievements?", "tell me more about your clients"]
  },
  "what are your achievements?": {
    "response": "We have received over 30 global awards, including Silver A'Design Winner and Digital Interaction Award. Our achievements showcase our commitment to excellence in design.",
    "options": ["tell me more about your work"]
  },
  "tell me more about your work": {
    "response": "We have designed projects like a fantasy trading platform and led a full-scale website redesign for SBI Caps. Our focus is on creating beautiful and functional experiences.",
    "options": ["what industries do you work in?"]
  },
  "what industries do you work in?": {
    "response": "We operate in various industries such as Healthcare and Wellness, Fintech and Banking, Agritech and Supply Chain, and Enterprise and Logistics.",
    "options": [""]
  },
  "bye": {
    "response": "Goodbye! Have a great day!",
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
  
  if (matchedOptions.length > 0) {
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
      addMessage("bot", botResponse.response);

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
    const response = await fetch('http://10.20.100.18:5000/ask', { // Change to your API endpoint
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        user_input: userText,
        section_id: sectionId, // Pass the section ID with the request
        model_choice:"ini"
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