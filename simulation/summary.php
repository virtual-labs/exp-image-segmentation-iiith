<?php

session_start();
echo '<head>
<script class='gtm'>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);})(window,document,'script','dataLayer','GTM-W59SWTR');</script>
';
echo ' <link rel="stylesheet" type="text/css" href="css/jquery-slider.css">';
echo '<link rel="stylesheet" type="text/css" href="css/psd2css.css" media="screen" />';
echo ' <script type="text/javascript" src="js/jquery-1.4.2.min.js"></script>
<script type="text/javascript"> 
jQuery.fn.popUpImg = function() {
    var o = $(this[0]) 
$(".imageFull","#Layer-1").css("top",($(this).offset()).top+"px");
    $("#fullImg",$(".imageFull","#Layer-1")).attr("src",$("img",$(o)).attr("src"));
    $(".imageFull").show("slow");
};

     $(document).ready (function () {
     $(".imageFull").hide();
     		 $(".item").each(function() {
    		 $(this).click(function() {
			$(this).popUpImg();
			});
  });});
</script>';
echo '</head>';
echo '<body>
  <div id="Layer-1" class="Backgound_bkgnd_center_jpg">
    <div id="Layer-3" class="TopBar_jpg"  >
      <img src="images/Layer-3.jpg" width="894" height="96" alt="TopBar" />
      <div id="Layer-6" class="IIIT"  >
        <img src="images/iiit.png" width="100" height="70" alt="IIIT" class="pngimg" />
      </div>
        <div id="topMenu">
<center>        <h1>VIRTUAL LAB in IMAGE PROCESSING</h1>
<input type="button" value="Close" onclick="window.close()" /></center>
        </div>
    </div>
    <div class="wrapper">';

if($_SESSION["state"]>0) { 
 	include ("temp/".$_SESSION["id"].".html");
} else {
  echo '<h2> Please perform some experiments before using the "Summary Button"</h2>';
}

echo '     <div class="imageFull">
 <div id="closeBut" style="position: absolute; right: 15px; top: 5px;">
	     <img src="images/close.png" onclick="$(\'.imageFull\').hide(\'slow\'); return false;"> 
	 </div>
		

     <div style="max-height: 680px; max-width: 680px; overflow: auto;"><img id="fullImg" href="#""> </div>
	</div>


</div></body>';

?>
