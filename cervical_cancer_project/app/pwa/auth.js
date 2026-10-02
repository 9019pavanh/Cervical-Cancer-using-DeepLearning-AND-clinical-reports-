const loginTab = document.getElementById('login-tab');
const registerTab = document.getElementById('register-tab');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const message = document.getElementById('auth-message');
const previewMode = location.pathname.startsWith('/preview/');
const routePrefix = previewMode ? '/preview' : '';

document.querySelectorAll('[data-route]').forEach(link => {
  const route = link.dataset.route;
  link.href = route ? `${routePrefix}/${route}` : (previewMode ? '/preview' : '/');
});

if (previewMode) document.getElementById('preview-notice').hidden = false;

function setMessage(text, type = '') {
  message.textContent = text;
  message.className = `auth-message${type ? ` ${type}` : ''}`;
}

function setMode(mode) {
  const register = mode === 'register';
  loginForm.hidden = register;
  registerForm.hidden = !register;
  loginTab.classList.toggle('active', !register);
  registerTab.classList.toggle('active', register);
  loginTab.setAttribute('aria-selected', String(!register));
  registerTab.setAttribute('aria-selected', String(register));
  history.replaceState(null, '', `${routePrefix}/${register ? 'register' : 'login'}`);
  setMessage('');
}

async function submit(url, payload, form) {
  if (!form.reportValidity()) return;
  if (previewMode) {
    setMessage('This is a visual preview. Open the live sign-in page to connect to PostgreSQL.', 'success');
    return;
  }
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  setMessage('Connecting securely…');
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Authentication failed');
    window.location.assign('/dashboard');
  } catch (error) {
    const detail = error.message.includes('DATABASE_URL') || error.message.includes('PostgreSQL')
      ? 'Backend database is not ready. Start PostgreSQL and configure DATABASE_URL.'
      : error.message;
    setMessage(detail, 'error');
  } finally {
    button.disabled = false;
  }
}

loginTab.addEventListener('click', () => setMode('login'));
registerTab.addEventListener('click', () => setMode('register'));

loginForm.addEventListener('submit', event => {
  event.preventDefault();
  submit('/api/auth/login', {
    email: document.getElementById('login-email').value,
    password: document.getElementById('login-password').value
  }, loginForm);
});

registerForm.addEventListener('submit', event => {
  event.preventDefault();
  const password = document.getElementById('register-password').value;
  if (password !== document.getElementById('register-password-confirm').value) {
    setMessage('Passwords do not match.', 'error');
    return;
  }
  submit('/api/auth/register', {
    name: document.getElementById('register-name').value,
    email: document.getElementById('register-email').value,
    password,
    role: document.getElementById('register-role').value
  }, registerForm);
});

setMode(location.pathname.endsWith('/register') ? 'register' : 'login');
