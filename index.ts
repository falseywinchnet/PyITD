// Import necessary modules
// Replace the itty-router import with a simple router implementation
const router = {
  routes: new Map(),
  get(path: string, handler: Function) {
    this.routes.set(`GET:${path}`, handler);
    return this;
  },
  post(path: string, handler: Function) {
    this.routes.set(`POST:${path}`, handler);
    return this;
  },
  all(path: string, handler: Function) {
    this.routes.set(`ALL:${path}`, handler);
    return this;
  },
  handle(request: Request, env: Env, ctx: ExecutionContext) {
    const url = new URL(request.url);
    const method = request.method;
    const path = url.pathname;
    
    // Check for exact route match
    const handler = this.routes.get(`${method}:${path}`) || this.routes.get(`ALL:${path}`);
    
    if (handler) {
      return handler(request, env, ctx);
    }
    
    // Check for wildcard match
    const wildcardHandler = this.routes.get('ALL:*');
    if (wildcardHandler) {
      return wildcardHandler(request, env, ctx);
    }
    
    return new Response('Not Found', { status: 404 });
  }
};
// Define your environment bindings
export interface Env {
  DB: D1Database;
}

// Define routes
router.get('/', handleMainPage);
router.post('/', handleFormSubmission);
router.post('/update', handleUpdateActivity);
router.all('*', () => new Response('Not Found', { status: 404 }));

// Configuration
const DEBUG = True; // Set to false in production
const SITE_TITLE = “website “title;

// Define question types
type Question = {
  type: string;
  text?: string;
  key: string;
  index: number;
  options?: string[];
  conditions?: string[];
};

// Questions definition
const QUESTIONS: Question[] = [
  {
    type: "CRIT",
    text: “Do you like pickles?”,
    key: "1", 
    index: 0
  },
  {
    type: "TEST",
    text: “What is 1 + 1?”,
    options: [
      “1”,
      “2”,
      “3”,
      “4”
    ],
    key: "1",
    index: 1
  },
  {
    type: "FORM",
    conditions: [
      “Yes”,
      “No“,
      “yes,
      “Yes”
    ],
    key: “1,0,1,1”,
    index: 2
  }
];

// HTML Templates
const TEMPLATE_SHELL = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${SITE_TITLE}</title>
  <style>
    /* Win98/Vaporwave/Watercolor Aesthetic */
    :root {
      --win98-gray: #c0c0c0;
      --win98-blue: #000080;
      --pastel-pink: #ffccdc;
      --pastel-blue: #ccf2ff;
      --pastel-purple: #e0ccff;
      --pastel-yellow: #ffffcc;
    }
    
    body {
      font-family: "MS Sans Serif", Arial, sans-serif;
      background: linear-gradient(135deg, var(--pastel-blue), var(--pastel-purple));
      margin: 0;
      padding: 0;
      color: #333;
      height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      background-attachment: fixed;
    }
    
    #app-container {
      width: 90%;
      max-width: 600px;
      background-color: var(--win98-gray);
      border: 2px solid #000;
      border-radius: 0;
      box-shadow: inset 1px 1px 0px #fff, 
                  inset -1px -1px 0px #888,
                  5px 5px 0 rgba(0,0,0,0.2);
      overflow: hidden;
    }
    
    .title-bar {
      background-color: var(--win98-blue);
      color: white;
      padding: 4px 8px;
      font-weight: bold;
      display: flex;
      justify-content: space-between;
    }
    
    .title-bar-controls {
      display: flex;
    }
    
    .title-bar-button {
      width: 16px;
      height: 14px;
      background-color: var(--win98-gray);
      border: 1px solid #000;
      margin-left: 4px;
      box-shadow: inset 1px 1px 0px #fff, inset -1px -1px 0px #888;
    }
    
    .content-area {
      padding: 20px;
      background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAEklEQVQImWNgYGD4z0AswK4SAFXuAf8EPy+xAAAAAElFTkSuQmCC');
      background-color: #f9f9f9;
      min-height: 300px;
    }
    
    .window-status-bar {
      background-color: var(--win98-gray);
      border-top: 1px solid #888;
      padding: 3px 8px;
      display: flex;
      justify-content: space-between;
      font-size: 12px;
    }
    
    /* Form Styling */
    .form-section {
      margin-bottom: 20px;
      background: white;
      padding: 15px;
      border: 1px solid #888;
      box-shadow: inset 1px 1px 0px #fff;
    }
    
    label {
      display: block;
      margin-bottom: 8px;
      font-weight: bold;
    }
    
    input[type="text"],
    input[type="email"] {
      width: 100%;
      padding: 8px;
      margin-bottom: 15px;
      border: 2px inset #aaa;
      background: white;
    }
    
    .radio-option {
      display: block;
      margin: 8px 0;
    }
    
    .btn-next {
      background-color: var(--win98-gray);
      border: 2px outset #ddd;
      padding: 5px 15px;
      font-family: inherit;
      cursor: pointer;
      float: right;
      color: black;
    }
    
    .btn-next:active {
      border-style: inset;
    }
    
    .form-table {
      width: 100%;
      border-collapse: collapse;
    }
    
    .form-table th, .form-table td {
      padding: 8px;
      border: 1px solid #ddd;
      text-align: left;
    }
    
    .form-table th {
      background-color: var(--pastel-blue);
    }
    
    /* Animation for transitions */
    .fade-in {
      animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }

    .radio-option input[type="radio"] {
      appearance: none;
      -webkit-appearance: none;
      width: 22px;
      height: 22px;
      margin-right: 12px;
      border: 2px solid #999;
      border-radius: 50%;
      background: radial-gradient(circle, #fff, var(--pastel-yellow));
      box-shadow: 0 0 4px rgba(255, 255, 255, 0.8);
      cursor: pointer;
      position: relative;
      top: 3px;
      transition: all 0.2s ease-in-out;
    }
    
    .radio-option input[type="radio"]:hover {
      box-shadow: 0 0 6px rgba(255, 182, 193, 0.9), inset 0 0 2px #fff;
      transform: scale(1.05);
    }
    
    .radio-option input[type="radio"]:checked {
      background: radial-gradient(circle, #ffd6e0, #ffccf2);
      border-color: #ff99cc;
      box-shadow: 0 0 8px #ffb3d9;
    }
    
    .radio-option label {
      font-size: 1.1em;
      padding-left: 6px;
      vertical-align: middle;
      cursor: pointer;
    }
    
  </style>
</head>
<body>
  <div id="app-container">
    <div class="title-bar">
      <div class="title-text">${SITE_TITLE}</div>
      <div class="title-bar-controls">
        <div class="title-bar-button">_</div>
        <div class="title-bar-button">□</div>
        <div class="title-bar-button">✕</div>
      </div>
    </div>
    <div class="content-area" id="content">
      <!-- Content will be injected here -->
    </div>
    <div class="window-status-bar">
      <div>Ready</div>
      <div id="progress-indicator">Question 0/0</div>
    </div>
  </div>
  
  <script>
    // Form handling script will be injected
  </script>
</body>
</html>`;

const TEMPLATE_START = `
<div class="form-section fade-in">


  <h1>Girlfriend Application</h1>
  <p> This is a test</p>

  
  <p>Please start by providing your name and some form of contact information below.</p>
  
  <div class="contact-form">
    <label for="name">Your Name:</label>
    <input type="text" id="name" name="name" required>
    
    <label for="email">Email Address(optional):</label>
    <input type="email" id="email" name="email">
    
    <label for="telegram">Telegram Handle (optional):</label>
    <input type="text" id="telegram" name="telegram">
    
    <label for="twitter">Twitter/X Handle (optional):</label>
    <input type="text" id="twitter" name="twitter">
    
    <button class="btn-next" id="start-survey">Begin Survey →</button>
  </div>
</div>`;

const TEMPLATE_END_SUCCESS = `
<div class="form-section fade-in">
  <h1>Congratulations!</h1>
  
  <p>good outcome</p>
  
  
</div>`;

const TEMPLATE_END_FAILURE = `
<div class="form-section fade-in">
  <h1>Thanks for Participating!</h1>
  
  <p>Bad outcome</p>
</div>`;

// Handler functions
async function handleMainPage(request: Request, env: Env) {
  const sessionId = generateSessionId();
  return new Response(generateFullPage(sessionId), {
    headers: {
      'Content-Type': 'text/html;charset=UTF-8',
    },
  });
}

async function handleFormSubmission(request: Request, env: Env) {
  let formData;
  try {
    formData = await request.json();
  } catch (error) {
    return new Response(JSON.stringify({ 
      success: false, 
      error: 'Invalid form data',
      html: '<div class="form-section">Error processing form data.</div>' 
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  // Check if DEBUG mode is active
  if (DEBUG) {
    return new Response(JSON.stringify({ 
      success: true, 
      html: TEMPLATE_END_SUCCESS 
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  // Evaluate answers against criteria
  const compatible = evaluateCompatibility(formData);
  
  // Store submission in D1
  if (!DEBUG) {
    try {
      await storeSubmission(formData, compatible, env.DB);
    } catch (error) {
      console.error('Error storing submission:', error);
      // Continue anyway - don't tell the user if storage failed
    }
  }
  
  // Return appropriate response
  return new Response(JSON.stringify({ 
    success: true, 
    html: compatible ? TEMPLATE_END_SUCCESS : TEMPLATE_END_FAILURE
  }), {
    headers: { 'Content-Type': 'application/json' }
  });
}

async function handleUpdateActivity(request: Request, env: Env) {
  try {
    const data = await request.json();
    
    if (!data.sessionId) {
      return new Response(JSON.stringify({ success: false }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    const now = Math.floor(Date.now() / 1000);
    
    // Update last activity timestamp
    await env.DB.prepare(`
      INSERT INTO submissions (
        session_id, last_activity, answers, created_at
      ) VALUES (?, ?, ?, ?)
      ON CONFLICT(session_id) DO UPDATE 
      SET last_activity = ?, answers = ?
    `).bind(
      data.sessionId,
      now,
      JSON.stringify(data.answers || {}),
      now,
      now,
      JSON.stringify(data.answers || {})
    ).run();
    
    return new Response(JSON.stringify({ success: true }), {
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Error updating activity:', error);
    return new Response(JSON.stringify({ success: false }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

// Helper functions
function generateFullPage(sessionId?: string): string {
  if (!sessionId) {
    sessionId = generateSessionId();
  }
  
  const jsScript = generateFormJS(sessionId);
  const htmlWithScript = TEMPLATE_SHELL.replace('// Form handling script will be injected', jsScript);
  
  return htmlWithScript;
}

function generatePlaceholderHTML(): string {
  return '<div class="form-section">Loading question...</div>';
}

function generateSessionId(): string {
  return 'xxxx-xxxx-xxxx-xxxx'.replace(/x/g, () => {
    return Math.floor(Math.random() * 16).toString(16);
  });
}

async function storeSubmission(formData: any, compatible: boolean, DB: D1Database) {
  const now = Math.floor(Date.now() / 1000);
  const sessionId = formData.sessionId || generateSessionId();
  
  try {
    // Store in D1 database
    await DB.prepare(`
      INSERT INTO submissions (
        session_id, name, email, telegram, twitter, 
        start_time, last_activity, completed, compatible, answers, created_at
      ) 
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      sessionId,
      formData.name || '',
      formData.email || '',
      formData.telegram || '',
      formData.twitter || '',
      now, // start_time
      now, // last_activity
      1,   // completed
      compatible ? 1 : 0,
      JSON.stringify(formData), // store all answers as JSON
      now  // created_at
    ).run();
    
    return true;
  } catch (error) {
    console.error('D1 error:', error);
    throw new Error(`D1 request failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function evaluateCompatibility(formData: any): boolean {
  let allCorrect = true;
  
  for (const question of QUESTIONS) {
    const userAnswer = formData['q' + question.index];
    
    if (!userAnswer) {
      allCorrect = false;
      break;
    }
    
    if (question.type === "CRIT" || question.type === "TEST") {
      if (userAnswer[0] !== question.key) {
        allCorrect = false;
        break;
      }
    } else if (question.type === "FORM") {
      const expectedAnswers = question.key.split(',');
      if (userAnswer.length !== expectedAnswers.length) {
        allCorrect = false;
        break;
      }
      
      for (let i = 0; i < expectedAnswers.length; i++) {
        if (userAnswer[i] !== expectedAnswers[i]) {
          allCorrect = false;
          break;
        }
      }
      
      if (!allCorrect) break;
    }
  }
  
  return allCorrect;
}

function generateFormJS(sessionId: string): string {
  return `
    // Form state handling
    let currentQuestion = -1; // Start screen
    const totalQuestions = ${QUESTIONS.length};
    let userAnswers = {};
    let sessionId = "${sessionId}"; // Injected from server
    
    // Get contact form fields
    const contactFields = ["name", "email", "telegram", "twitter"];
    
    // Question data for client-side rendering
    const questions = ${JSON.stringify(QUESTIONS)};
    
    // Debug mode
    const DEBUG = ${DEBUG};
    
    // Activity tracking
    let lastActivityTime = Date.now();
    const ACTIVITY_TIMEOUT = 20 * 60 * 1000; // 20 minutes
    
    document.addEventListener('click', updateActivity);
    document.addEventListener('keydown', updateActivity);
    
    function updateActivity() {
      lastActivityTime = Date.now();
      
      // Send update to server if we're in a survey
      if (currentQuestion >= 0) {
        fetch('/update', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            sessionId: sessionId,
            currentQuestion: currentQuestion,
            answers: userAnswers
          })
        }).catch(error => {
          console.error('Error updating activity:', error);
        });
      }
    }
    
    // Update progress indicator
    function updateProgress() {
      if (currentQuestion >= 0) {
        document.getElementById('progress-indicator').textContent = 
          \`Question \${currentQuestion + 1}/\${totalQuestions}\`;
      } else {
        document.getElementById('progress-indicator').textContent = 'Contact Info';
      }
    }
    
    // Navigation functions
    function showStartScreen() {
      currentQuestion = -1;
      document.getElementById('content').innerHTML = \`${TEMPLATE_START}\`;
      document.getElementById('start-survey').addEventListener('click', () => {
        // Validate contact form
        let valid = true;
        if (!DEBUG) {
          const name = document.getElementById('name').value.trim();
          const email = document.getElementById('email').value.trim();
          const telegram = document.getElementById('telegram').value.trim();
          const twitter = document.getElementById('twitter').value.trim();
        
          if (!name) {
            alert('Please enter your name');
            valid = false;
          }
        
          if (!email && !telegram && !twitter) {
            alert('Please provide at least one contact method (email, Telegram, or Twitter)');
            valid = false;
          }
        }
        
        if (valid) {
          // Save contact info
          contactFields.forEach(field => {
            userAnswers[field] = document.getElementById(field).value.trim();
          });
          
          // Show first question
          showQuestion(0);
        }
      });
      updateProgress();
    }
    
    function showQuestion(index) {
      if (index >= totalQuestions) {
        submitForm();
        return;
      }
      
      currentQuestion = index;
      const question = questions[index];
      
      document.getElementById('content').innerHTML = \`${generatePlaceholderHTML()}\`;
      document.getElementById('content').innerHTML = generateQuestionHTML(question);
      
      // Add event listener to the next button
      document.querySelector('.btn-next').addEventListener('click', () => {
        // Validate question
        let valid = true;
        let answers = [];
        
        if (!DEBUG) {
          if (question.type === "CRIT") {
            const selected = document.querySelector(\`input[name="q\${question.index}"]:checked\`);
            if (!selected) {
              alert('Please select an answer');
              valid = false;
            } else {
              answers.push(selected.value);
            }
          } else if (question.type === "TEST") {
            const selected = document.querySelector(\`input[name="q\${question.index}"]:checked\`);
            if (!selected) {
              alert('Please select an answer');
              valid = false;
            } else {
              answers.push(selected.value);
            }
          } else if (question.type === "FORM") {
            for (let i = 0; i < question.conditions.length; i++) {
              const selected = document.querySelector(\`input[name="q\${question.index}_\${i}"]:checked\`);
              if (!selected) {
                alert('Please answer all statements');
                valid = false;
                break;
              } else {
                answers.push(selected.value);
              }
            }
          }
        } else {
          // In debug mode, generate fake answers
          if (question.type === "CRIT" || question.type === "TEST") {
            answers.push("1");
          } else if (question.type === "FORM") {
            for (let i = 0; i < question.conditions.length; i++) {
              answers.push("1");
            }
          }
        }
        
        if (valid) {
          // Save answers
          userAnswers['q' + question.index] = answers;
          
          // Show next question
          showQuestion(index + 1);
        }
      });
      
      updateProgress();
    }
    
    function generateQuestionHTML(question) {
      let html = '<div class="form-section fade-in">';
      
      if (question.type === "CRIT") {
        html += \`
          <h3>\${question.text}</h3>
          <div class="radio-option">
            <input type="radio" id="q\${question.index}_yes" name="q\${question.index}" value="1" required>
            <label for="q\${question.index}_yes">Yes</label>
          </div>
          <div class="radio-option">
            <input type="radio" id="q\${question.index}_no" name="q\${question.index}" value="0">
            <label for="q\${question.index}_no">No</label>
          </div>
        \`;
      } else if (question.type === "TEST") {
        html += \`<h3>\${question.text}</h3>\`;
        
        question.options.forEach((option, idx) => {
          html += \`
            <div class="radio-option">
              <input type="radio" id="q\${question.index}_\${idx}" name="q\${question.index}" value="\${idx}" required>
              <label for="q\${question.index}_\${idx}">\${option}</label>
            </div>
          \`;
        });
      } else if (question.type === "FORM") {
        html += \`
          <h3>Please indicate which statements apply to you:</h3>
          <table class="form-table">
            <tr>
              <th>Statement</th>
              <th>Applies</th>
              <th>Doesn't Apply</th>
            </tr>
        \`;
        
        question.conditions.forEach((condition, idx) => {
          html += \`
            <tr>
              <td>\${condition}</td>
              <td><input type="radio" name="q\${question.index}_\${idx}" value="1" required></td>
              <td><input type="radio" name="q\${question.index}_\${idx}" value="0"></td>
            </tr>
          \`;
        });
        
        html += '</table>';
      }
      
      html += \`<button class="btn-next" data-question="\${question.index}">Next →</button>\`;
      html += '</div>';
      
      return html;
    }
    
    async function submitForm() {
      // Include session ID in submission
      userAnswers.sessionId = sessionId;
      
      // In real mode, send data to server
      if (!DEBUG) {
        try {
          const response = await fetch('/', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(userAnswers)
          });
          
          const data = await response.json();
          document.getElementById('content').innerHTML = data.html;
        } catch (error) {
          console.error('Error submitting form:', error);
          document.getElementById('content').innerHTML = '<div class="form-section">Error submitting form. Please try again.</div>';
        }
      } 
      
      document.getElementById('progress-indicator').textContent = 'Complete';
    }
    
    // Initialize the form
    showStartScreen();
  `;
}

// Export the fetch handler function as the default export
export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    return router.handle(request, env, ctx);
  }
};
