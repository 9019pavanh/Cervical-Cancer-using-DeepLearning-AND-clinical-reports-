const previewMode = location.pathname.startsWith('/preview');

document.querySelectorAll('[data-route]').forEach(link => {
  const route = link.dataset.route;
  link.href = previewMode ? `/preview/${route}`.replace(/\/$/, '') : `/${route}`;
});

document.querySelectorAll('[data-preview-dashboard]').forEach(link => {
  link.href = '/preview/dashboard';
});

if ('serviceWorker' in navigator && !previewMode) {
  navigator.serviceWorker.register('/service-worker.js').catch(() => {});
}
