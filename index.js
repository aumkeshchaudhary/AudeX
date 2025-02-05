// Sidebar functions
function openNav() {
    document.getElementById("mySidebar").style.width = "250px";
}

function closeNav() {
    document.getElementById("mySidebar").style.width = "0";
}

// Section navigation
function showSection(section) {
    const sections = document.querySelectorAll('section');
    sections.forEach(s => s.classList.remove('active'));
    document.getElementById(section).classList.add('active');
}

// Login/Signup toggle
function toggleLoginSignup(form) {
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    if (form === 'signup') {
        loginForm.style.display = 'none';
        signupForm.style.display = 'block';
    } else {
        loginForm.style.display = 'block';
        signupForm.style.display = 'none';
    }
}

// Handle image upload and processing using Tesseract.js
async function processImage() {
    const fileInput = document.getElementById('fileInput');
    const language = document.getElementById('languageSelect').value;
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = async function(event) {
            const img = new Image();
            img.src = event.target.result;
            img.onload = async function() {
                Tesseract.recognize(
                    img,
                    language,
                    {
                        logger: m => console.log(m)
                    }
                ).then(({ data: { text } }) => {
                    document.getElementById('outputText').value = text;
                });
            }
        };
        reader.readAsDataURL(file);
    } else {
        alert('Please upload an image first.');
    }
}

// Handle text-to-speech functionality
function playText() {
    const text = document.getElementById('outputText').value;
    if (text) {
        const speech = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(speech);
    } else {
        alert('No text to read.');
    }
}

function stopText() {
    window.speechSynthesis.cancel();
}

function downloadTextAsPDF() {
    const text = document.getElementById('outputText').value;
    if (text) {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        doc.text(text, 10, 10);
        doc.save('text.pdf');
    } else {
        alert('No text to download.');
    }
}

function countWordsAndCharacters() {
    const text = document.getElementById('outputText').value;
    const words = text.trim().split(/\s+/).length;
    const characters = text.length;
    alert(`Words: ${words}, Characters: ${characters}`);
}

function searchKeyword() {
    const text = document.getElementById('outputText').value;
    const keyword = document.getElementById('keywordInput').value;
    if (keyword) {
        const regex = new RegExp(keyword, 'gi');
        const matches = text.match(regex);
        alert(`Found ${matches ? matches.length : 0} occurrences of "${keyword}".`);
    } else {
        alert('Please enter a keyword to search.');
    }
}

// Contact us form
function contact() {
    alert('You can contact us at support@audex.com');
}

// Save profile edits
function saveProfile() {
    const name = document.getElementById('editName').value;
    const address = document.getElementById('editAddress').value;
    const phone = document.getElementById('editPhone').value;
    const country = document.getElementById('editCountry').value;
    const dob = document.getElementById('editDob').value;

    // Placeholder logic for saving the profile
    document.getElementById('profileName').innerText = name || 'John Doe';
    document.getElementById('profileAddress').innerText = address || '123 Main Street, Some City';
    document.getElementById('profilePhone').innerText = phone || '+91-1234567890';
    document.getElementById('profileCountry').innerText = country || 'India';
    document.getElementById('profileLastLogin').innerText = dob || '2025-01-07';

    cancelEditProfile();
}

// Cancel profile edit
function cancelEditProfile() {
    document.getElementById('profileEdit').style.display = 'none';
    document.getElementById('profileView').style.display = 'block';
}

// Toggle profile edit mode
function toggleEditProfile() {
    const profileView = document.getElementById('profileView');
    const profileEdit = document.getElementById('profileEdit');
    if (profileEdit.style.display === 'none') {
        profileEdit.style.display = 'block';
        profileView.style.display = 'none';
        // Populate edit fields with current values
        document.getElementById('editName').value = document.getElementById('profileName').innerText;
        document.getElementById('editAddress').value = document.getElementById('profileAddress').innerText;
        document.getElementById('editPhone').value = document.getElementById('profilePhone').innerText;
        document.getElementById('editCountry').value = document.getElementById('profileCountry').innerText;
        document.getElementById('editDob').value = document.getElementById('profileLastLogin').innerText;
    } else {
        profileEdit.style.display = 'none';
        profileView.style.display = 'block';
    }
}

// Delete history function
function deleteHistory() {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '';
}

function login() {
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;

    if (email && password) {
        alert("Login successful! Welcome to Audex.");
        showSection('profile');
    } else {
        alert("Please fill in both fields.");
    }
}

function signup() {
    const name = document.getElementById("signupName").value;
    const email = document.getElementById("signupEmail").value;
    const password = document.getElementById("signupPassword").value;
    const rePassword = document.getElementById("signupRePassword").value;
    const country = document.getElementById("signupCountry").value;
    const dob = document.getElementById("signupDob").value;
    const address = document.getElementById("signupAddress").value;
    const phone = document.getElementById("signupPhone").value;

    if (password !== rePassword) {
        alert("Passwords do not match!");
        return;
    }

    // Placeholder logic for signing up
    alert("Sign up successful! Please log in.");
    showSection('login');
}

function formatText() {
    const input = document.getElementById('formatterInput').value;
    const template = document.getElementById('formatterTemplate').value;
    // Placeholder logic for formatting text based on template
    document.getElementById('formatterOutput').value = `Formatted (${template}): ${input}`;
}

function convertToBulletPoints() {
    const input = document.getElementById('conversionInput').value;
    const output = input.split('\n').map(line => `• ${line}`).join('\n');
    document.getElementById('conversionOutput').value = output;
}

function convertToNumberedList() {
    const input = document.getElementById('conversionInput').value;
    const output = input.split('\n').map((line, index) => `${index + 1}. ${line}`).join('\n');
    document.getElementById('conversionOutput').value = output;
}

function convertToTable() {
    const input = document.getElementById('conversionInput').value;
    const rows = input.split('\n').map(line => `<tr><td>${line}</td></tr>`).join('');
    const output = `<table border="1">${rows}</table>`;
    document.getElementById('conversionOutput').value = output;
}

function changeCase(caseType) {
    const input = document.getElementById('conversionInput').value;
    let output;
    switch (caseType) {
        case 'upper':
            output = input.toUpperCase();
            break;
        case 'lower':
            output = input.toLowerCase();
            break;
        case 'sentence':
            output = input.charAt(0).toUpperCase() + input.slice(1).toLowerCase();
            break;
    }
    document.getElementById('conversionOutput').value = output;
}

function transliterateText() {
    const input = document.getElementById('transliterationInput').value;
    // Placeholder logic for transliteration
    document.getElementById('transliterationOutput').value = `Transliterated: ${input}`;
}

function generateCreativeContent() {
    const input = document.getElementById('creativeInput').value;
    // Placeholder logic for generating creative content
    document.getElementById('creativeOutput').value = `Generated Content: ${input}`;
}

function createPrintDesign() {
    const input = document.getElementById('printDesignerInput').value;
    // Placeholder logic for creating print design
    document.getElementById('printDesignOutput').innerHTML = `<div style="font-size: 24px; color: #000;">${input}</div>`;
}

function generateAsciiArt() {
    const input = document.getElementById('playgroundInput').value;
    // Placeholder logic for generating ASCII art
    document.getElementById('playgroundOutput').value = `ASCII Art: ${input}`;
}

function createWordCloud() {
    const input = document.getElementById('playgroundInput').value;
    // Placeholder logic for creating word cloud
    document.getElementById('playgroundOutput').value = `Word Cloud: ${input}`;
}

function encodeText() {
    const input = document.getElementById('playgroundInput').value;
    // Placeholder logic for encoding text
    document.getElementById('playgroundOutput').value = `Encoded: ${btoa(input)}`;
}

function decodeText() {
    const input = document.getElementById('playgroundInput').value;
    // Placeholder logic for decoding text
    document.getElementById('playgroundOutput').value = `Decoded: ${atob(input)}`;
}
