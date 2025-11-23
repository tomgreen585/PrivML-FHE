//wait until DOM loaded before executing nav logic
document.addEventListener('DOMContentLoaded', () => {
  //select all dropdown buttons, dropdown content blocks, and nav links
  const dropButtons = document.querySelectorAll('.dropbtn');
  const dropdowns = document.querySelectorAll('.dropdown-content');
  const navLinks = document.querySelectorAll('.nav-item');

  //removes active class from nav links and dropdown buttons
  function clearActive() {
    navLinks.forEach(a => a.classList.remove('active'));
    dropButtons.forEach(b => b.classList.remove('active'));
  }

  //adds active class based on specific key (home, tools, github)
  function setActiveKey(key) {
    clearActive();
    if (key === 'home') {
      const home = document.querySelector('.nav-item[href="/"]');
      if (home) home.classList.add('active');
    } else if (key === 'tools') {
      const toolsBtn = document.querySelector('.dropbtn[data-dropdown="tools"]');
      if (toolsBtn) toolsBtn.classList.add('active');
    } else if (key === 'github') {
      const gBtn = document.querySelector('.dropbtn[data-dropdown="github"]');
      if (gBtn) gBtn.classList.add('active');
    }
  }

  //detect current page based on pathname and set correct active state
  const path = location.pathname || '/';
  if (path === '/' || path === '/index.html') {
    setActiveKey('home');
  } else if (
    path.startsWith('/border_ml') ||
    path.startsWith('/face_ml')   ||
    path.startsWith('/mnist_ml')
  ) {
    setActiveKey('tools');
  } else {
    //if research was going to be added later
  }

  //set up dropdown button click behavior
  dropButtons.forEach(button => {
    button.addEventListener('click', (e) => {
      e.stopPropagation(); //prevent click from propagating document listener
      
      //identify the dropdown menu linked to this button
      const id = button.getAttribute('data-dropdown');
      const menu = document.getElementById(`dropdown-${id}`);
      
      //close all other dropdowns except the one being toggled
      dropdowns.forEach(dc => { if (dc !== menu) dc.style.display = 'none'; });
      
      //toggle clicked dropdown
      const willOpen = menu.style.display !== 'block';
      menu.style.display = willOpen ? 'block' : 'none';
      
      //highlight the active tab when dropdown is opened
      if (willOpen) {
        if (id === 'tools') setActiveKey('tools');
        else if (id === 'github') setActiveKey('github');
      }
    });
  });

  //close all dropdowns when clicking somewhere else on page
  document.addEventListener('click', () => {
    dropdowns.forEach(dc => dc.style.display = 'none');
  });

  //add handler to ensure it becomes active on click
  const homeLink = document.querySelector('.nav-item[href="/"]');
  if (homeLink) {
    homeLink.addEventListener('click', () => setActiveKey('home'));
  }
  
});
