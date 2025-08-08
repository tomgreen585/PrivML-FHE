document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.dropbtn').forEach(button => {
    button.addEventListener('click', function (e) {
      e.stopPropagation();

      const dropdownId = this.getAttribute('data-dropdown');
      const dropdown = document.getElementById('dropdown-' + dropdownId);

      document.querySelectorAll('.dropdown-content').forEach(dc => {
        if (dc !== dropdown) dc.style.display = 'none';
      });

      dropdown.style.display = dropdown.style.display === 'block' ? 'none' : 'block';
    });
  });

  document.addEventListener('click', () => {
    document.querySelectorAll('.dropdown-content').forEach(dc => {
      dc.style.display = 'none';
    });
  });
});
