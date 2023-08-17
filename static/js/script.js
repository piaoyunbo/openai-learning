document.addEventListener("DOMContentLoaded", () => {
  const inputElement = document.getElementById("input");
  const submitButton = document.getElementById("submit-button");
//   inputElement.addEventListener("keydown", (event) => {
//       if (event.key === "Enter") {
//           event.preventDefault();
//           const userInput = inputElement.value.trim();
//           if (userInput) {
//               addMessage(userInput, "user-message");
//               inputElement.value = "";
//               fetchResponse(userInput);
//           }
//       }
//   });
  submitButton.addEventListener("click", () => {
      const userInput = inputElement.value.trim();
      if (userInput) {
          addMessage(userInput, "user-message");
          inputElement.value = "";
          fetchResponse(userInput);
      }
  });
});

function addMessage(content, className) {
  const chatbox = document.getElementById("chatbox");
  const messageWrapper = document.createElement("div");
  messageWrapper.classList.add(className);

  const message = document.createElement("div");
  message.classList.add("message");
  if (content instanceof Array) {
    content.forEach(element => {
      const p = document.createElement("p");
      p.textContent = element.trim();
      message.appendChild(p);
    });
  } else {
    message.textContent = content;
  }

  messageWrapper.appendChild(message);
  chatbox.appendChild(messageWrapper);

  if (className === "user-message") {
      messageWrapper.style.justifyContent = "flex-end";
  }

  chatbox.scrollTop = chatbox.scrollHeight;
}

async function fetchResponse(userInput) {
  // Replace with your Azure Base URL
  const hostname = location.hostname;
  const azureBaseUrl = `http://${hostname}:15000/app/`;

  try {
      const response = await fetch(azureBaseUrl, {
          method: "POST",
          headers: {
              "Content-Type": "application/json",
          },
        //   body: JSON.stringify({ prompt: `user\n${userInput}\n\nassistant` }),
          body: JSON.stringify({ user_input: `${userInput}` }),
      });

      if (response.ok) {
          const responseData = await response.json();
          const assistantResponse = responseData.response.trim();
          const responseMsg = String(assistantResponse).split("\n");
          addMessage(responseMsg, "assistant-message");
      } else {
          console.error("Error fetching response:", response.status, response.statusText);
          addMessage("An error has occurred. Please try again.", "assistant-message");
      }
  } catch (error) {
      console.error("Error fetching response:", error);
      addMessage("A communication error has occurred. Please try again.", "assistant-message");
  }
}